/*
 * ======================================================================
 * esp32_har_streamer.ino
 * ======================================================================
 * Project: Human Activity Recognition (HAR) using ESP32 + MPU-6050
 * Description: 
 *   High-speed 50Hz IMU Telemetry Streamer with Dual USB-Serial (115200)
 *   and Bluetooth Low Energy (BLE) sensor broadcasting.
 *
 * Hardware Pinout (ESP32 NodeMCU):
 *   - MPU-6050 VCC -> 3.3V (or 5V if module has onboard regulator)
 *   - MPU-6050 GND -> GND
 *   - MPU-6050 SDA -> GPIO 21
 *   - MPU-6050 SCL -> GPIO 22
 *   - MPU-6050 INT -> GPIO 19 (Optional hardware interrupt)
 *   - Status LED   -> GPIO 2  (Built-in LED, pulses on sampling)
 *
 * Sampling Frequency: 50 Hz (20 ms interval)
 * Output Format (CSV): Ax,Ay,Az,Gx,Gy,Gz,Timestamp_ms\n
 *
 * 
 * ======================================================================
 */

#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// MPU-6050 I2C Address
#define MPU6050_ADDR 0x68

// Sampling configuration
#define SAMPLING_FREQ_HZ 50
#define SAMPLING_PERIOD_US (1000000 / SAMPLING_FREQ_HZ) // 20,000 us (20 ms)

// Pin Definitions
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define STATUS_LED_PIN 2

// BLE Configuration
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

// Calibration Offsets
float accX_offset = 0.0, accY_offset = 0.0, accZ_offset = 0.0;
float gyroX_offset = 0.0, gyroY_offset = 0.0, gyroZ_offset = 0.0;

// Timing variables
unsigned long previousMicros = 0;
uint32_t packetCount = 0;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
    }
};

void readRawMPU(int16_t &ax, int16_t &ay, int16_t &az, int16_t &gx, int16_t &gy, int16_t &gz) {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x3B); // ACCEL_XOUT_H register
    Wire.endTransmission(false);
    Wire.requestFrom(MPU6050_ADDR, 14, true);

    ax = (Wire.read() << 8 | Wire.read());
    ay = (Wire.read() << 8 | Wire.read());
    az = (Wire.read() << 8 | Wire.read());
    int16_t tempRaw = (Wire.read() << 8 | Wire.read()); // Temperature (unused)
    gx = (Wire.read() << 8 | Wire.read());
    gy = (Wire.read() << 8 | Wire.read());
    gz = (Wire.read() << 8 | Wire.read());
}

void setupMPU6050() {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, 400000); // 400 kHz Fast I2C
    
    // Wake up MPU-6050 (write 0 to PWR_MGMT_1 register 0x6B)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x6B);
    Wire.write(0x00);
    Wire.endTransmission(true);
    
    // Configure Accelerometer (+/- 2g range: write 0x00 to ACCEL_CONFIG register 0x1C)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x1C);
    Wire.write(0x00);
    Wire.endTransmission(true);

    // Configure Gyroscope (+/- 250 deg/s range: write 0x00 to GYRO_CONFIG register 0x1B)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x1B);
    Wire.write(0x00);
    Wire.endTransmission(true);

    // Digital Low-Pass Filter Configuration (DLPF_CFG = 3 -> 44Hz filter bandwidth)
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x1A);
    Wire.write(0x03);
    Wire.endTransmission(true);
}

void calibrateSensors() {
    digitalWrite(STATUS_LED_PIN, HIGH);
    Serial.println("# Calibrating MPU-6050... Keep board stationary on flat surface.");
    
    const int samples = 200;
    int16_t ax, ay, az, gx, gy, gz;
    long a_x = 0, a_y = 0, a_z = 0, g_x = 0, g_y = 0, g_z = 0;

    for (int i = 0; i < samples; i++) {
        readRawMPU(ax, ay, az, gx, gy, gz);
        a_x += ax; a_y += ay; a_z += az;
        g_x += gx; g_y += gy; g_z += gz;
        delay(5);
    }

    // Gravity is 1.0g along Z when flat (16384 LSB/g)
    accX_offset = (float)a_x / samples / 16384.0;
    accY_offset = (float)a_y / samples / 16384.0;
    accZ_offset = ((float)a_z / samples / 16384.0) - 1.0; 
    gyroX_offset = (float)g_x / samples / 131.0;
    gyroY_offset = (float)g_y / samples / 131.0;
    gyroZ_offset = (float)g_z / samples / 131.0;

    Serial.printf("# Calibrated: AccOffset(%.3f, %.3f, %.3f), GyroOffset(%.3f, %.3f, %.3f)\n",
                  accX_offset, accY_offset, accZ_offset, gyroX_offset, gyroY_offset, gyroZ_offset);
    digitalWrite(STATUS_LED_PIN, LOW);
}

void setupBLE() {
    BLEDevice::init("ESP32-HAR-Streamer");
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    BLEService *pService = pServer->createService(SERVICE_UUID);
    pCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID,
                        BLECharacteristic::PROPERTY_READ   |
                        BLECharacteristic::PROPERTY_WRITE  |
                        BLECharacteristic::PROPERTY_NOTIFY |
                        BLECharacteristic::PROPERTY_INDICATE
                      );
    pCharacteristic->addDescriptor(new BLE2902());
    pService->start();

    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(false);
    pAdvertising->setMinPreferred(0x0);
    BLEDevice::startAdvertising();
    Serial.println("# BLE Active: 'ESP32-HAR-Streamer' advertising started.");
}

void setup() {
    Serial.begin(115200);
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);

    delay(1000);
    Serial.println("\n# =========================================================");
    Serial.println("# ESP32 HAR 50Hz IMU Streamer - Deep Learning Bridge");
    Serial.println("# =========================================================");

    setupMPU6050();
    calibrateSensors();
    setupBLE();

    Serial.println("# HEADER: Ax(g),Ay(g),Az(g),Gx(rad/s),Gy(rad/s),Gz(rad/s),Timestamp_ms");
    previousMicros = micros();
}

void loop() {
    unsigned long currentMicros = micros();

    // Enforce 50Hz precision timing loop (every 20,000 microseconds)
    if (currentMicros - previousMicros >= SAMPLING_PERIOD_US) {
        previousMicros += SAMPLING_PERIOD_US;
        packetCount++;

        int16_t raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz;
        readRawMPU(raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz);

        // Convert to standard units: Accel in g, Gyro in radians/sec (standard UCI HAR units)
        float ax = ((float)raw_ax / 16384.0) - accX_offset;
        float ay = ((float)raw_ay / 16384.0) - accY_offset;
        float az = ((float)raw_az / 16384.0) - accZ_offset;

        // Convert deg/s to rad/s (deg/s * PI / 180)
        float gx = (((float)raw_gx / 131.0) - gyroX_offset) * (PI / 180.0);
        float gy = (((float)raw_gy / 131.0) - gyroY_offset) * (PI / 180.0);
        float gz = (((float)raw_gz / 131.0) - gyroZ_offset) * (PI / 180.0);

        unsigned long now_ms = millis();

        // 1. Output high-throughput CSV over USB Serial
        char buffer[128];
        snprintf(buffer, sizeof(buffer), "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%lu\n",
                 ax, ay, az, gx, gy, gz, now_ms);
        Serial.print(buffer);

        // 2. Broadcast over Bluetooth Low Energy if client connected
        if (deviceConnected && (packetCount % 2 == 0)) { // 25Hz BLE updates to save bandwidth
            pCharacteristic->setValue((uint8_t*)buffer, strlen(buffer));
            pCharacteristic->notify();
        }

        // Heartbeat LED flash every 1 second (50 packets)
        if (packetCount % 50 == 0) {
            digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
        }
    }
}
