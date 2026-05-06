#include "Adafruit_MLX90393.h"

Adafruit_MLX90393 sensor = Adafruit_MLX90393();
#define MLX90393_CS 10

float hard_iron_bias_x = -21.443491;
float hard_iron_bias_y = -39.953870;
float hard_iron_bias_z = 16.793674;

double soft_iron_bias_xx = 1.011895;
double soft_iron_bias_xy = 0.012423;
double soft_iron_bias_xz = -0.028616;

double soft_iron_bias_yx = 0.012423;
double soft_iron_bias_yy = 1.026209;
double soft_iron_bias_yz = 0.019366;

double soft_iron_bias_zx = -0.028616;
double soft_iron_bias_zy = 0.019366;
double soft_iron_bias_zz = 1.071255;

// Function to apply calibration to raw magnetometer data
void applyCalibration(float raw_x, float raw_y, float raw_z, 
                      float &cal_x, float &cal_y, float &cal_z) {
    // Step 1: Remove hard iron offset
    float hard_x = raw_x - hard_iron_bias_x;
    float hard_y = raw_y - hard_iron_bias_y;
    float hard_z = raw_z - hard_iron_bias_z;
    
    // Step 2: Apply soft iron transformation matrix
    cal_x = (soft_iron_bias_xx * hard_x) + (soft_iron_bias_xy * hard_y) + (soft_iron_bias_xz * hard_z);
    cal_y = (soft_iron_bias_yx * hard_x) + (soft_iron_bias_yy * hard_y) + (soft_iron_bias_yz * hard_z);
    cal_z = (soft_iron_bias_zx * hard_x) + (soft_iron_bias_zy * hard_y) + (soft_iron_bias_zz * hard_z);
}

void setup(void)
{
  Serial.begin(115200);

  /* Wait for serial on USB platforms. */
  while (!Serial) {
      delay(10);
  }

  Serial.println("Starting Adafruit MLX90393 Calibrated Demo");

  if (! sensor.begin_I2C()) {
    Serial.println("No sensor found ... check your wiring?");
    while (1) { delay(10); }
  }
  Serial.println("Found a MLX90393 sensor");

  sensor.setGain(MLX90393_GAIN_2X);
  
  Serial.print("Gain set to: ");
  switch (sensor.getGain()) {
    case MLX90393_GAIN_1X: Serial.println("1 x"); break;
    case MLX90393_GAIN_1_33X: Serial.println("1.33 x"); break;
    case MLX90393_GAIN_1_67X: Serial.println("1.67 x"); break;
    case MLX90393_GAIN_2X: Serial.println("2 x"); break;
    case MLX90393_GAIN_2_5X: Serial.println("2.5 x"); break;
    case MLX90393_GAIN_3X: Serial.println("3 x"); break;
    case MLX90393_GAIN_4X: Serial.println("4 x"); break;
    case MLX90393_GAIN_5X: Serial.println("5 x"); break;
  }

  // Set resolution
  sensor.setResolution(MLX90393_X, MLX90393_RES_17);
  sensor.setResolution(MLX90393_Y, MLX90393_RES_17);
  sensor.setResolution(MLX90393_Z, MLX90393_RES_16);

  // Set oversampling
  sensor.setOversampling(MLX90393_OSR_3);

  // Set digital filtering
  sensor.setFilter(MLX90393_FILTER_5);
  
  Serial.println("Calibration values loaded");
}

void loop(void) {
    float raw_x, raw_y, raw_z;
    float cal_x, cal_y, cal_z;
    
    // get raw X Y and Z data
    if (sensor.readData(&raw_x, &raw_y, &raw_z)) {
        
        // Apply hard iron and soft iron calibration
        applyCalibration(raw_x, raw_y, raw_z, cal_x, cal_y, cal_z);
        // cal_z = raw_z*1.235;
        // cal_x= raw_x*1.118;
        // cal_y = raw_y*0.874;

        
        // Check for invalid values
        if (isnan(cal_x) || isnan(cal_y) || isnan(cal_z) || 
            isinf(cal_x) || isinf(cal_y) || isinf(cal_z)) {
            Serial.println("Error - Invalid calibrated values");
        } else {
            // Print using simple Serial.print instead of snprintf
            Serial.print("M:");
            Serial.print(cal_x, 4);
            Serial.print(",");
            Serial.print(cal_y, 4);
            Serial.print(",");
            Serial.println(cal_z, 4);
        }
    } else {
        Serial.println("Error - Unable to read XYZ data from the sensor");
    }
    
}