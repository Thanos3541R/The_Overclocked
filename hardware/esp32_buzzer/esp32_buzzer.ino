// The Overclocked - ESP32 DevKit V1 Active Buzzer HIL Controller
// Wiring: ESP32 3V3 -> Buzzer VCC(+), ESP32 GND -> Buzzer GND(-), ESP32 D4 (GPIO4) -> Buzzer I/O(S)
#define BUZZER_PIN 4

// Most 3-pin Active Buzzers are Active-LOW (LOW = ON, HIGH = OFF).
// If your buzzer rings continuously when idle ('0'), swap LOW and HIGH below.
#define BUZZER_ON  HIGH
#define BUZZER_OFF LOW

char currentState = '0';  // '0' = SOBER, '1' = WARN, '2' = LOCKOUT / MICROSLEEP
unsigned long lastBeepToggle = 0;
bool beepState = false;

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, BUZZER_OFF);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '0' || c == '1' || c == '2') {
      currentState = c;
    }
  }

  unsigned long now = millis();
  if (currentState == '2') {
    digitalWrite(BUZZER_PIN, BUZZER_ON);
  } else if (currentState == '1') {
    if (now - lastBeepToggle > (beepState ? 200 : 300)) {
      beepState = !beepState;
      digitalWrite(BUZZER_PIN, beepState ? BUZZER_ON : BUZZER_OFF);
      lastBeepToggle = now;
    }
  } else {
    digitalWrite(BUZZER_PIN, BUZZER_OFF);
    beepState = false;
  }
  delay(10);
}
