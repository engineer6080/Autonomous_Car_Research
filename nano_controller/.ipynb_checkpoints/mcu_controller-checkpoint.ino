#include "ServoTimer2.h"
#define INA 3
#define INB 4
#define DC_PWM 5
#define SERVO_PWM 6
#define SPEED_PIN 2

#define WHEELCIRC  (PI * 0.055)
// Diameter = 57.5 mm or ~ 5.5cm
// Centimeters / sec RANGE: 0 to 30cm sec ~ 1 feet

ServoTimer2 servo;

void setup()
{
   
    // Setup Serial Monitor
     Serial.begin(1000000); //115200

     pinMode(DC_PWM, OUTPUT);
     servo.attach(SERVO_PWM);

     pinMode(INA, OUTPUT);
     pinMode(INB, OUTPUT);
  
     //pinMode(SPEED_PIN, INPUT);
     attachInterrupt(digitalPinToInterrupt(SPEED_PIN), rpm_int, FALLING);

     digitalWrite(INA, HIGH);
     digitalWrite(INB, LOW);
  
     //Base Frequency PIN DC 5 = 62500
     //Base Frequency PIN SERVO 6 = 62500
     // 312.5 ~ 312 for 200hz

      //setPwmFrequency(5, 312); D5, D6
      //TCCR0B = TCCR0B & B11111000 | B00000100; // for PWM frequency of 244.14 Hz
}

int dc_out = 0;
int servo_out = 1500;

const byte numChars = 12;
char receivedChars[numChars];   // an array to store the received data
boolean newData = false;

volatile int revolutions = 0;
unsigned long lastInterrupt = millis();

unsigned int rpm = 0;
unsigned long timeold = 0;
unsigned long k = 0;
double car_speed = 0;

double P = 0.0002;
double pid_out = 0;
double dc_temp = 0;

void rpm_int() {
  if(lastInterrupt-millis()>10){
    revolutions++;
    lastInterrupt = millis();
  }  
}


void loop()
{   
      recvWithEndMarker();
      if(newData == true){      
            int servo_raw = 90;
            int dc_raw = 90;

            int n = sscanf((char*)receivedChars, "%5d-%5d", &dc_raw, &servo_raw);
            servo_out = map(servo_raw, 0, 180, 2000, 1000);

            if(dc_raw >=0){
                  int dc_temp = map(dc_raw, 0, 180, 0, 255); 
                  //0-125 reverse; 135-256 forward; else coast
                  if(dc_temp >= 138){
                        digitalWrite(INA, HIGH);
                        digitalWrite(INB, LOW);
                        dc_out = dc_temp - 128;
                  }
                  else if (dc_temp < 100){
                        digitalWrite(INA, LOW);
                        digitalWrite(INB, HIGH);
                        dc_out = 128-dc_temp;
                  }
                  else{
                        digitalWrite(INA, LOW);
                        digitalWrite(INB, LOW);
                  }
            }
            else{ //AUTO THROTTLE CONTROL
               //int dc_temp = map(abs(dc_raw), 0, 180, 0, 255); 
               int desired_speed = abs(dc_raw); // input in cm/s
               double p_err = (desired_speed - car_speed); //-300 to 100
               pid_out = (p_err * P);
               int max_speed = 80; //0 to 128
               dc_temp = max(min((pid_out+dc_temp),max_speed),50);
               
               /*
               Serial.print(p_err);
               Serial.print("\t");
               Serial.print(pid_out);
               Serial.print("\t");
               Serial.print(dc_temp);
               Serial.print("\t");
               Serial.println(rpm);
			   */
               
              dc_out = dc_temp;
              if (pid_out <= -0.01){
                  //if(pid_out <= -1){
                    digitalWrite(INA, LOW);
                    digitalWrite(INB, LOW);
                    //delay(10);
                  //}
                  digitalWrite(INA, HIGH);
                  digitalWrite(INB, LOW);
              }
              else{
                  digitalWrite(INA, HIGH);
                  digitalWrite(INB, LOW);
              }
      
        }// else

    k = (millis()-timeold);
    if(revolutions >= 9){ //increase for accuracy (4 rotations) revolutions >= 28 ||
      detachInterrupt(0); //Interrupts are disabled
      rpm = ((double)(revolutions/3)/k)*60*1000; //RPM
      car_speed = rpm;
      timeold = millis();
      revolutions = 0;
      attachInterrupt(digitalPinToInterrupt(SPEED_PIN), rpm_int, FALLING);
    }
    Serial.println(rpm);
    newData = false;
  }//new Data
  
    servo.write(servo_out);
    analogWrite(DC_PWM, dc_out);
}//loop

void recvWithEndMarker() {
    static byte ndx = 0;
    char endMarker = '\n';
    char rc;
   
    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        if (rc != endMarker) {
            receivedChars[ndx] = rc;
            ndx++;
            if (ndx >= numChars) {
                ndx = numChars - 1;
            }
        }
        else {
            receivedChars[ndx] = '\0'; // terminate the string
            ndx = 0;
            newData = true;
        }
    }
}
