#include <Wire.h>
#include <LiquidCrystal_I2C.h>  
#include <MQ135.h>
#include "DHT.h"

#define DHTPIN 2
#define DHTTYPE DHT11
#define MIST_PIN 3
#define PIN_MQ135 A0

DHT dht(DHTPIN, DHTTYPE);
MQ135 mq135_sensor(PIN_MQ135, 330, 22);
LiquidCrystal_I2C lcd(0x27, 16, 2); // I2C地址0x27，16x2 LCD

int page = 0;                           // 當前顯示頁面
unsigned long prevPageMillis = 0;
const long pageInterval = 10000;        // 10秒換頁

void setup() {
  Serial.begin(9600);
  dht.begin();
  pinMode(MIST_PIN, OUTPUT);

  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.print("Env Monitor Demo");
  delay(2000);
  lcd.clear();
}

void loop() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (isnan(h) || isnan(t)) {
    Serial.println("感測失敗！");
    return;
  }

  // 霧化控制
  if (h < 75) {
    digitalWrite(MIST_PIN, LOW);
  } else {
    digitalWrite(MIST_PIN, HIGH);
  }

  // MQ-135 讀值
  float correctedppm = mq135_sensor.getCorrectedPPM(t, h);

  // 10秒自動換頁
  if (millis() - prevPageMillis >= pageInterval) {
    page = (page + 1) % 2;
    prevPageMillis = millis();
    lcd.clear();  // 換頁時清屏
  }
 
  static unsigned long lastResetMillis = 0;
  if (millis() - lastResetMillis >= 300000) { // 每5分鐘重置
  float new_rzero = mq135_sensor.getCorrectedRZero(t, h);
  mq135_sensor = MQ135(PIN_MQ135, new_rzero, 22);
  lastResetMillis = millis();
  }
  // 顯示內容
  lcd.setCursor(0, 0);
  if (page == 0) {
    // 第一頁：溫度、濕度
    lcd.print("Temp: ");
    lcd.print(t, 1);
    lcd.print(" C");
    lcd.setCursor(0, 1);
    lcd.print("Humid: ");
    lcd.print(h, 1);
    lcd.print(" %");
  } else {
    // 第二頁：corrected ppm
    lcd.print("MQ135 PPM: ");
    lcd.setCursor(0, 1);
    lcd.print(correctedppm, 1);
    lcd.print(" ppm");
  }

  // 串口輸出（可選）
  Serial.print("\n濕度: ");
  Serial.print(h);
  Serial.print(" %\t");
  Serial.print("溫度: ");
  Serial.print(t);
  Serial.println(" *C");
  Serial.print("Corrected PPM: ");
  Serial.println(correctedppm);

  delay(1000);
}
