/*
 * ======================================================================
 * arduino_mpu6050.ino
 * ======================================================================
 * Project: Human Activity Recognition (HAR) using Arduino Uno/Nano + MPU-6050
 * Description: 
 *   Optimized lightweight 50Hz IMU Telemetry Streamer for standard 
 *   Arduino microcontrollers (ATmega328P).
 *
 * Hardware Pinout (Arduino Uno / Nano):
 *   - MPU-6050 VCC -> 5V (or 3.3V)
 *   - MPU-6050 GND -> GND
 *   - MPU-6050 SDA -> A4 (SDA)
 *   - MPU-6050 SCL -> A5 (SCL)
 *   - Status LED   -> Pin 13 (Built-in LED)
 *
 * Baud Rate: 115200 bps
 * Output CSV Format: Ax,Ay,Az,Gx,Gy,Gz,Timestamp_ms\n
 *
 * 
 * ======================================================================
 */

#include <Wire.h>

#define MPU6050_ADDR 0x68
#define SAMPLING_PERIOD_MS 20  // 50 Hz = 20ms

const float PI_VAL = 3.14159265359;
unsigned long nextSampleTime = 0;

// Offsets
float ax_off = 0, ay_off = 0, az_off = 0;
float gx_off = 0, gy_off = 0, gz_off = 0;

void setup() {
    Serial.begin(115200);
    pinMode(13, OUTPUT);
    digitalWrite(13, LOW);

    Wire.begin();
    TWBR = 12; // 400kHz fast I2C mode for ATmega328P

    // Wake up MPU-6050
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x6B);
    Wire.write(0);
    Wire.endTransmission(true);

    // Accel range +/- 2g (16384 LSB/g)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x1C);
    Wire.write(0x00);
    Wire.endTransmission(true);

    // Gyro range +/- 250 deg/s (131 LSB/deg/s)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x1B);
    Wire.write(0x00);
    Wire.endTransmission(true);

    // Calibration routine (100 samples)
    digitalWrite(13, HIGH);
    long a_x = 0, a_y = 0, a_z = 0, g_x = 0, g_y = 0, g_z = 0;
    for (int i = 0; i < 100; i++) {
        Wire.beginTransmission(MPU6050_ADDR);
        Wire.write(0x3B);
        Wire.endTransmission(false);
        Wire.requestFrom(MPU6050_ADDR, 14, true);

        int16_t ax = Wire.read() << 8 | Wire.read();
        int16_t ay = Wire.read() << 8 | Wire.read();
        int16_t az = Wire.read() << 8 | Wire.read();
        Wire.read(); Wire.read(); // Skip temp
        int16_t gx = Wire.read() << 8 | Wire.read();
        int16_t gy = Wire.read() << 8 | Wire.read();
        int16_t gz = Wire.read() << 8 | Wire.read();

        a_x += ax; a_y += ay; a_z += az;
        g_x += gx; g_y += gy; g_z += gz;
        delay(10);
    }
    ax_off = (float)a_x / 100.0 / 16384.0;
    ay_off = (float)a_y / 100.0 / 16384.0;
    az_off = ((float)a_z / 100.0 / 16384.0) - 1.0;
    gx_off = (float)g_x / 100.0 / 131.0;
    gy_off = (float)g_y / 100.0 / 131.0;
    gz_off = (float)g_z / 100.0 / 131.0;
    digitalWrite(13, LOW);

    Serial.println("# READY: Arduino MPU-6050 50Hz Streamer");
    nextSampleTime = millis();
}

void loop() {
    unsigned long now = millis();
    if (now >= nextSampleTime) {
        nextSampleTime += SAMPLING_PERIOD_MS;

        Wire.beginTransmission(MPU6050_ADDR);
        Wire.write(0x3B);
        Wire.endTransmission(false);
        Wire.requestFrom(MPU6050_ADDR, 14, true);

        int16_t raw_ax = Wire.read() << 8 | Wire.read();
        int16_t raw_ay = Wire.read() << 8 | Wire.read();
        int16_t raw_az = Wire.read() << 8 | Wire.read();
        Wire.read(); Wire.read(); // Skip temp
        int16_t raw_gx = Wire.read() << 8 | Wire.read();
        int16_t raw_gy = Wire.read() << 8 | Wire.read();
        int16_t raw_gz = Wire.read() << 8 | Wire.read();

        float ax = ((float)raw_ax / 16384.0) - ax_off;
        float ay = ((float)raw_ay / 16384.0) - ay_off;
        float az = ((float)raw_az / 16384.0) - az_off;
        float gx = (((float)raw_gx / 131.0) - gx_off) * (PI_VAL / 180.0);
        float gy = (((float)raw_gy / 131.0) - gy_off) * (PI_VAL / 180.0);
        float gz = (((float)raw_gz / 131.0) - gz_off) * (PI_VAL / 180.0);

        Serial.print(ax, 4); Serial.print(",");
        Serial.print(ay, 4); Serial.print(",");
        Serial.print(az, 4); Serial.print(",");
        Serial.print(gx, 4); Serial.print(",");
        Serial.print(gy, 4); Serial.print(",");
        Serial.print(gz, 4); Serial.print(",");
        Serial.println(now);
    }
}
