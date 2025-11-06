import re
import json
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter


class SmartSpaceAgentChatbot:
    """
    SOS: Space Agent - Intelligent chatbot with multi-layer understanding
    - Keyword matching with fuzzy logic
    - Context awareness
    - Intent classification
    - Entity extraction
    - Fallback mechanisms
    """

    def __init__(self, dataset_path: str = None):
        self.agent_name = "NOVA"
        self.mission_status = "ACTIVE"
        self.conversation_history = []
        self.user_name = None
        self.stress_level = 0
        self.last_topic = None
        self.awaiting_clarification = False
        self.clarification_options = []

        # Load comprehensive dataset
        self.dataset = self.load_default_dataset()

        # Load custom dataset if provided
        if dataset_path:
            self.load_dataset(dataset_path)

        # Build keyword index for fast matching
        self.build_keyword_index()

    def load_default_dataset(self) -> List[Dict]:
        """Load comprehensive default dataset"""
        return [
            # OXYGEN EMERGENCIES
            {
                "keywords": [
                    "oxygen",
                    "o2",
                    "air",
                    "breathing",
                    "suffocate",
                    "breath",
                    "atmosphere",
                    "venting",
                    "leak",
                    "pressure",
                ],
                "response": """🚨 OXYGEN EMERGENCY PROTOCOL - CRITICAL

⚠️ IMMEDIATE ACTIONS (Next 60 seconds):
1. Don EMV suit or EVA suit IMMEDIATELY
2. Activate personal O2 supply - verify gauge >500 PSI
3. Alert all crew - sound general alarm
4. Move to safe module through sealed hatch

🔧 IF OXYGEN LEAK:
• Close isolation valves to affected area
• Activate emergency O2 reserves
• Use handheld detector to find breach
• Patch with emergency sealant kit

🔧 IF LOW OXYGEN (No leak):
• Check life support systems
• Replace CO2 scrubber canisters
• Verify oxygen generator functioning
• Switch to backup O2 supply

📊 CRITICAL LEVELS:
• Normal: 21% O2
• Safe minimum: 19.5% O2
• Dangerous: <19% O2
• Time to unconsciousness at 10%: 30-60 seconds

Survival with reserves: 24-48 hours

What's your current O2 reading? Any visible damage?""",
                "severity": "CRITICAL",
                "category": "life_support",
                "questions": [
                    "What is your current oxygen level reading?",
                    "Can you see any visible leaks or damage?",
                    "How many crew members are affected?",
                ],
            },
            # FIRE EMERGENCIES
            {
                "keywords": [
                    "fire",
                    "flame",
                    "smoke",
                    "burning",
                    "burn",
                    "combustion",
                    "blaze",
                    "ignite",
                    "spark",
                ],
                "response": """🚨 FIRE EMERGENCY PROTOCOL - CRITICAL

⚠️ IMMEDIATE ACTIONS (Next 30 seconds):
1. Sound fire alarm - evacuate area NOW
2. Don breathing apparatus
3. Close ALL ventilation (starves fire)
4. Activate CO2 suppression in affected zone

🧯 FIRE SUPPRESSION:
• Use CO2 extinguishers ONLY (never water!)
• Cut electrical power to area
• Seal compartment with bulkhead doors
• Monitor adjacent areas for spread

⚠️ WARNING:
• Fire doubles every 30 seconds in oxygen
• Toxic smoke spreads in 60 seconds
• DO NOT fight if flames >1 meter

📊 POST-SUPPRESSION:
• Keep sealed 30+ minutes
• Check for reignition with thermal camera
• Verify no electrical shorts
• Assess structural damage

🔍 Common causes: Electrical shorts (45%), batteries (30%), equipment (25%)

What started the fire? How large is it? Electrical/Chemical/Other?""",
                "severity": "CRITICAL",
                "category": "fire",
                "questions": [
                    "What was the source of the fire?",
                    "How large are the flames?",
                    "Is it electrical, chemical, or other?",
                ],
            },
            # HULL BREACH
            {
                "keywords": [
                    "hull",
                    "breach",
                    "hole",
                    "puncture",
                    "depressurization",
                    "decompression",
                    "pressure",
                    "vacuum",
                    "crack",
                    "rupture",
                    "meteor",
                    "micrometeorite",
                ],
                "response": """🚨 HULL BREACH / DEPRESSURIZATION - CRITICAL

⚠️ IMMEDIATE (15 seconds - before consciousness loss):
1. SUITS ON NOW!
2. Activate suit pressure - check seal
3. Secure yourself (prevent ejection)
4. Hit emergency beacon

🔍 LOCATE BREACH (Next 60 seconds):
• Listen for whistling/rushing air
• Use pressure detector
• Feel for airflow with glove
• Deploy foam sealant (<5cm holes)
• Use hull patch kit (>5cm)

📊 TIME CRITICAL:
• 1cm hole: 3 minutes to danger
• 5cm hole: 45 seconds to danger
• >10cm: EVACUATE IMMEDIATELY (10 seconds)

🛡️ TEMPORARY FIXES:
• Duct tape + plastic sheet: 5-10 min
• Stuff with fabric/foam
• Use airlock as refuge

⚡ SUIT OXYGEN:
• EVA Suit: 6-8 hours (rest), 2-4 hours (active)
• Emergency Vest: 30-45 minutes ONLY

Breach size? Current pressure reading?""",
                "severity": "CRITICAL",
                "category": "structural",
                "questions": [
                    "How large is the breach?",
                    "What's your current pressure reading?",
                    "Are you in a suit?",
                ],
            },
            # RADIATION
            {
                "keywords": [
                    "radiation",
                    "solar",
                    "flare",
                    "cosmic",
                    "rays",
                    "dosimeter",
                    "exposure",
                    "radioactive",
                ],
                "response": """🚨 RADIATION EXPOSURE PROTOCOL

⚠️ IMMEDIATE (2 minutes):
1. Move to radiation shelter (storm shelter) NOW
2. Close all window shutters
3. Activate magnetic shielding
4. Check dosimeter reading
5. Don radiation vest

📊 DOSIMETER LEVELS:
• <50 mSv: Safe (normal background)
• 50-100 mSv: Elevated - monitor
• 100-500 mSv: Moderate - shelter
• 500-1000 mSv: High - emergency
• >1000 mSv: SEVERE - immediate medical

☢️ SHIELDING:
• Position water tanks between you and source
• Use food/equipment as mass shielding
• Deploy polyethylene barriers
• Stay in spacecraft center

🌞 SOLAR FLARE:
• Warning time: 8-15 minutes
• Peak radiation: 2-6 hours after
• Safe to exit: 24-48 hours later

⚕️ SYMPTOMS TO WATCH:
• Nausea/vomiting (early sign)
• Take anti-radiation meds from kit
• Monitor every 30 minutes

Current dosimeter reading? Any symptoms?""",
                "severity": "HIGH",
                "category": "radiation",
                "questions": [
                    "What's your dosimeter reading?",
                    "Are you experiencing any symptoms?",
                    "Are you in the radiation shelter?",
                ],
            },
            # COMMUNICATION LOSS
            {
                "keywords": [
                    "communication",
                    "comms",
                    "radio",
                    "signal",
                    "antenna",
                    "contact",
                    "transmission",
                    "connection",
                    "reach",
                    "lost",
                ],
                "response": """📡 COMMUNICATION LOSS PROTOCOL

🔧 IMMEDIATE CHECKS (5 minutes):
1. Check main comm array status panel
2. Verify power (should be 28V DC)
3. Switch to backup array (toggle Alpha-3)
4. Test emergency beacon (121.5 MHz)
5. Try laser comm link

🛰️ ANTENNA TROUBLESHOOTING:
• Use star tracker for spacecraft orientation
• Point high-gain antenna toward Earth
• Use signal strength meter to fine-tune

📡 FREQUENCIES TO TRY:
• Primary: 2.2 GHz (S-band)
• Backup: 8.4 GHz (X-band)
• Emergency: 121.5 MHz (distress)

🔌 POWER CYCLE PROCEDURE:
1. Turn OFF main comms (30 sec)
2. Check/reset circuit breakers
3. Turn ON backup comms first
4. Wait 60 sec for boot
5. Turn ON main comms

📊 SIGNAL STRENGTH:
• Excellent: >-80 dBm
• Good: -80 to -100 dBm
• Weak: -100 to -120 dBm
• No signal: <-120 dBm

📝 BACKUP METHODS:
• Text-only mode (90% less bandwidth)
• Record messages for batch send
• Morse code on emergency frequency

What happened when comms failed? Unusual sounds/smells?""",
                "severity": "HIGH",
                "category": "communication",
                "questions": [
                    "When did you lose contact?",
                    "Any unusual events before loss?",
                    "Is the power indicator on?",
                ],
            },
            # POWER FAILURE
            {
                "keywords": [
                    "power",
                    "electrical",
                    "electricity",
                    "battery",
                    "solar",
                    "panels",
                    "energy",
                    "blackout",
                    "shutdown",
                    "voltage",
                ],
                "response": """⚡ POWER SYSTEM FAILURE PROTOCOL

🔋 IMMEDIATE ACTIONS:
1. Switch to battery backup (auto)
2. Reduce non-essential power NOW:
   ❌ Disable: Entertainment, non-critical lights, experiments
   ✅ Keep: Life support, navigation, comms, thermal

📊 EMERGENCY POWER BUDGET:
• Life support: 1.2 kW (CRITICAL)
• Thermal control: 0.8 kW (CRITICAL)
• Comms: 0.3 kW (CRITICAL)
• Navigation: 0.2 kW (CRITICAL)
TOTAL: 2.6 kW minimum

🔍 DIAGNOSIS:

SOLAR PANELS:
• Check deployment (180° extended?)
• Inspect for damage via cameras
• Verify sun tracking (within 2°)
• Clean if dust/debris present
• Expected: 4-14 kW based on sun distance

BATTERY:
• Check charge level (maintain >40%)
• Safe range: 20-80%
• If hot (>60°C): Disconnect immediately
• Lifetime: ~50,000 cycles

🛠️ TROUBLESHOOTING:
1. Reset power management computer
2. Check main bus voltage (28V ±4V)
3. Inspect circuit breakers
4. Test each source independently

⏰ BATTERY DURATION:
• Full ops: 2-4 hours
• Emergency mode: 12-24 hours
• Survival only: 48-72 hours

Current battery %? Solar panel status?""",
                "severity": "HIGH",
                "category": "power",
                "questions": [
                    "What's your battery level?",
                    "Are solar panels deployed?",
                    "Which systems failed first?",
                ],
            },
            # LIFE SUPPORT
            {
                "keywords": [
                    "life support",
                    "co2",
                    "carbon dioxide",
                    "scrubber",
                    "eclss",
                    "temperature",
                    "humidity",
                    "hot",
                    "cold",
                    "condensation",
                ],
                "response": """🌬️ LIFE SUPPORT SYSTEM FAILURE

⚠️ CHECK READINGS NOW:
• O2: 19.5-23% (CRITICAL if <19%)
• CO2: <0.5% (DANGEROUS if >1%)
• Pressure: 14.7 PSI
• Temp: 18-24°C
• Humidity: 30-70%

🔧 CO2 SCRUBBER FIX:

Lithium Hydroxide (LiOH):
1. Replace canister (in ECLSS bay)
2. Each lasts 2-person-days
3. Listen for airflow (slight hiss)
4. Check weight (+20% = saturated)

Molecular Sieve:
1. Check temp cycling (0-200°C)
2. Verify vacuum pump running
3. Listen for valve cycle (every 5 min)
4. Reset if frozen

🚨 EMERGENCY CO2 REMOVAL:
• Use backup canisters immediately
• Adapt EVA suit scrubbers (2-hour capacity)
• Use sodium hydroxide if LiOH gone
• Reduce activity (40% less CO2)

💨 OXYGEN GENERATION:
• Need 2L water/person/day
• Check voltage 2.0-2.4V
• Inspect hydrogen vent
• Clean electrode buildup

🌡️ TEMPERATURE:
• Check coolant pump circulation
• Verify radiator deployment
• Look for coolant leaks (UV light)
• Rising >2°C/hour = failure

⏰ SURVIVAL TIME:
• CO2 critical: 6-12 hours
• O2 depleted: 24-48 hours
• Temperature: 12-24 hours

What readings are abnormal? Which component?""",
                "severity": "CRITICAL",
                "category": "life_support",
                "questions": [
                    "What are your current atmospheric readings?",
                    "Which component seems to have failed?",
                    "How long has it been malfunctioning?",
                ],
            },
            # MEDICAL
            {
                "keywords": [
                    "medical",
                    "injured",
                    "hurt",
                    "sick",
                    "pain",
                    "bleeding",
                    "blood",
                    "unconscious",
                    "broken",
                    "fracture",
                    "burn",
                    "wound",
                    "emergency",
                ],
                "response": """🏥 MEDICAL EMERGENCY PROTOCOL

⚕️ IMMEDIATE ASSESSMENT:
1. Conscious? Alert? Responsive?
2. Breathing? Rate 12-20/min normal?
3. Pulse? 60-100/min normal?
4. Bleeding? Apply pressure
5. Move to medical bay if safe

🚑 SPECIFIC EMERGENCIES:

UNCONSCIOUS:
• Clear airway
• Recovery position (if no spinal injury)
• Monitor vitals every 5 min
• Prepare CPR

SEVERE BLEEDING:
• Direct pressure with sterile gauze
• Elevate above heart
• Tourniquet 5cm above wound (severe only)
• Use suction (microgravity)

FRACTURES:
• Immobilize with splint
• Ice pack (cold pack from kit)
• Pain relief: Ibuprofen
• Secure patient in microgravity

BURNS:
• Cool with water 10-20 min
• Sterile non-stick dressing
• DO NOT pop blisters
• Severe: evacuation needed

💊 MEDICAL KIT:
• Pain: Ibuprofen, acetaminophen
• Antibiotics: Amoxicillin
• Cardiac: Aspirin, epinephrine
• Suture kit, scalpel
• IV fluids & administration set

📡 TELEMEDICINE:
1. Activate medical downlink
2. Transmit vitals continuously
3. Video with flight surgeon
4. Follow guidance precisely

⏰ EVACUATION IF:
• Cardiac arrest
• Severe trauma
• Uncontrolled bleeding
• Unconscious >30 min
• Surgical needs

What are the symptoms? Patient conscious? Visible injuries?""",
                "severity": "HIGH",
                "category": "medical",
                "questions": [
                    "What are the patient's symptoms?",
                    "Is the patient conscious?",
                    "Are there any visible injuries?",
                ],
            },
            # NAVIGATION
            {
                "keywords": [
                    "navigation",
                    "lost",
                    "position",
                    "location",
                    "course",
                    "trajectory",
                    "guidance",
                    "gps",
                    "direction",
                    "heading",
                ],
                "response": """🧭 NAVIGATION SYSTEM FAILURE

📍 POSITION CHECK NOW:
1. Check IMU (inertial measurement)
2. Verify star tracker (needs 3+ stars)
3. Query GPS (Earth orbit - 4+ satellites)
4. Calculate from last known position

🛰️ BACKUP NAVIGATION:

STAR TRACKER:
1. Identify 3 bright stars:
   • Polaris (North Star) - mag 2.0
   • Sirius (brightest) - mag -1.5
   • Canopus (2nd brightest) - mag -0.7
2. Use star catalog for triangulation
3. Input angles to nav computer
Accuracy: ±0.1°

CELESTIAL NAVIGATION:
1. Use sextant from emergency kit
2. Measure Earth/Moon/Sun angles
3. Cross-ref ephemeris tables
4. Calculate with spherical trig
Accuracy: ±10-50 km

GROUND TRACKING:
• Request position from mission control
• Ground radar: ±1 km accuracy
• Signal delay reveals distance

🔧 SYSTEM RESET:
IMU: Power cycle (90 sec), calibrate gyros (5 min)
Star Tracker: Manual ID mode, clean sensor
GPS: Point toward Earth, cold start 10-15 min

🗺️ TRAJECTORY CORRECTION:
1. Calculate position error
2. Determine delta-V needed
3. Plan RCS burn timing
4. Execute & verify

📊 FUEL COSTS:
• Minor (10 m/s): 2-5 kg
• Moderate (50 m/s): 10-25 kg
• Major (100+ m/s): May be impossible

Position estimate? Mission phase? Fuel remaining?""",
                "severity": "HIGH",
                "category": "navigation",
                "questions": [
                    "What's your current position estimate?",
                    "What mission phase are you in?",
                    "How much fuel do you have remaining?",
                ],
            },
            # SPACE INFO
            {
                "keywords": ["mars", "red planet", "planet"],
                "response": """🔴 MARS - THE RED PLANET

📊 VITAL STATS:
• Distance: 228M km (1.52 AU from Sun)
• Day: 24.6 hours (like Earth!)
• Year: 687 Earth days
• Gravity: 38% of Earth
• Atmosphere: 95% CO2
• Pressure: 0.6% of Earth
• Temp: -60°C avg (-140 to +20°C)

🌡️ SURVIVAL CHALLENGES:
• Thin atmosphere - suit required always
• No magnetic field - radiation exposure
• Dust storms last months
• Extreme cold
• Water only as ice

🚀 HUMAN MISSIONS:
• Travel time: 6-9 months
• Launch window: Every 26 months
• Mission duration: 2-3 years total
• Typical crew: 4-6 astronauts

🔬 COOL FACTS:
• Olympus Mons: Largest volcano (24km tall!)
• Valles Marineris: 4,000km canyon
• Ancient riverbeds prove past water
• Possible microbial life in subsurface ice

Want to know about Mars missions or other planets?""",
                "severity": "INFO",
                "category": "astronomy",
                "questions": [],
            },
            # STRESS SUPPORT
            {
                "keywords": [
                    "stress",
                    "anxious",
                    "anxiety",
                    "worried",
                    "scared",
                    "afraid",
                    "lonely",
                    "depressed",
                    "isolated",
                    "panic",
                    "mental",
                    "psychological",
                ],
                "response": """🧠 PSYCHOLOGICAL SUPPORT

Agent, I recognize you're under stress. This is completely normal in space missions. You're not alone - these feelings don't mean weakness.

🌟 IMMEDIATE TECHNIQUES:

1. BREATHING (2 min):
   • Inhale 4 counts
   • Hold 4 counts
   • Exhale 6 counts
   • Repeat 5 times

2. GROUNDING (5-4-3-2-1):
   • 5 things you SEE
   • 4 things you TOUCH
   • 3 things you HEAR
   • 2 things you SMELL
   • 1 thing you TASTE

3. MUSCLE RELAXATION:
   • Tense each muscle 5 sec
   • Release and feel relaxation
   • Toes to head

💪 SPACE-SPECIFIC COPING:
• Schedule Earth video calls
• Join crew for meals
• Keep mission journal
• Exercise daily (mood booster)
• Earth viewing (15 min calms mind)
• Listen to favorite music

🏥 REQUEST HELP IF:
• Thoughts of self-harm
• Can't sleep 3+ days
• Unable to perform duties
• Panic attacks increasing
• Loss of appetite >48 hours

📞 SUPPORT AVAILABLE:
• Flight surgeon 24/7
• Confidential psych support
• No judgment - 60% of missions
• Medication available if needed

💙 REMEMBER:
You were chosen for exceptional capability. Every astronaut faces challenges. Asking for help is professional strength.

What's specifically concerning you? Or want a distraction?""",
                "severity": "INFO",
                "category": "psychological",
                "questions": [
                    "What specifically is worrying you?",
                    "Would you like to talk about it or prefer a distraction?",
                ],
            },
        ]

    def load_dataset(self, filepath: str):
        """Load custom dataset from JSON"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                custom_data = json.load(f)

            for entry in custom_data:
                if all(
                    key in entry
                    for key in ["keywords", "response", "severity", "category"]
                ):
                    self.dataset.append(entry)

            print(f"✅ Loaded {len(custom_data)} custom entries")
            self.build_keyword_index()

        except FileNotFoundError:
            print(f"❌ Error: File '{filepath}' not found")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    def build_keyword_index(self):
        """Build inverted index for fast keyword lookup"""
        self.keyword_index = {}

        for idx, entry in enumerate(self.dataset):
            for keyword in entry["keywords"]:
                keyword_lower = keyword.lower()
                if keyword_lower not in self.keyword_index:
                    self.keyword_index[keyword_lower] = []
                self.keyword_index[keyword_lower].append(idx)

    def fuzzy_keyword_match(self, user_input: str) -> List[Tuple[int, int]]:
        """
        Match user input against keywords with fuzzy matching
        Returns list of (dataset_index, match_score) tuples
        """
        user_words = re.findall(r"\b\w+\b", user_input.lower())
        matches = Counter()

        for word in user_words:
            # Exact match
            if word in self.keyword_index:
                for idx in self.keyword_index[word]:
                    matches[idx] += 10

            # Partial match (word contains keyword or vice versa)
            for keyword, indices in self.keyword_index.items():
                if len(word) >= 4 and len(keyword) >= 4:
                    if word in keyword or keyword in word:
                        for idx in indices:
                            matches[idx] += 5
                    # Check for common substring
                    elif self.longest_common_substring(word, keyword) >= 4:
                        for idx in indices:
                            matches[idx] += 3

        # Return sorted by score
        return [(idx, score) for idx, score in matches.most_common()]

    def longest_common_substring(self, s1: str, s2: str) -> int:
        """Find length of longest common substring"""
        m, n = len(s1), len(s2)
        max_len = 0

        for i in range(m):
            for j in range(n):
                k = 0
                while i + k < m and j + k < n and s1[i + k] == s2[j + k]:
                    k += 1
                max_len = max(max_len, k)

        return max_len

    def detect_emergency_intent(self, message: str) -> bool:
        """Detect if user is reporting an emergency"""
        emergency_indicators = [
            r"\b(emergency|urgent|help|sos|crisis|critical|mayday|911)\b",
            r"\b(dying|dead|death)\b",
            r"\b(can\'t|cannot|unable|won\'t|failing)\b",
            r"!{2,}",  # Multiple exclamation marks
        ]

        for pattern in emergency_indicators:
            if re.search(pattern, message.lower()):
                return True
        return False

    def extract_emergency_type(self, message: str) -> str:
        """Try to determine what type of emergency from context"""
        categories = {
            "oxygen/breathing": [
                "oxygen",
                "o2",
                "air",
                "breath",
                "suffocate",
                "atmosphere",
            ],
            "fire": ["fire", "flame", "smoke", "burning", "burn"],
            "hull breach": ["breach", "hole", "pressure", "depressurization", "vacuum"],
            "radiation": ["radiation", "solar", "flare", "dosimeter"],
            "power": ["power", "electrical", "battery", "energy"],
            "communication": ["comms", "communication", "radio", "signal", "antenna"],
            "medical": ["injured", "hurt", "sick", "pain", "bleeding", "unconscious"],
            "navigation": ["lost", "navigation", "position", "course"],
            "life support": ["life support", "co2", "temperature", "hot", "cold"],
        }

        message_lower = message.lower()
        detected = []

        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in message_lower:
                    detected.append(category)
                    break

        return detected

    def generate_response(self, message: str) -> str:
        """Main intelligence system with multiple fallback layers"""

        # Store in history
        self.conversation_history.append(
            {"role": "user", "message": message, "timestamp": datetime.now()}
        )

        message_lower = message.lower()

        # LAYER 1: Handle clarification responses
        if self.awaiting_clarification and self.clarification_options:
            # User is answering our question
            for i, option in enumerate(self.clarification_options):
                if str(i + 1) in message or option["category"].lower() in message_lower:
                    self.awaiting_clarification = False
                    response = option["response"]
                    self.last_topic = option["category"]
                    self.clarification_options = []

                    self.conversation_history.append(
                        {
                            "role": "assistant",
                            "message": response,
                            "timestamp": datetime.now(),
                        }
                    )
                    return response

        # LAYER 2: Basic intent detection
        # Greeting
        if re.search(r"\b(hi|hello|hey|greetings|sup)\b", message_lower):
            if not self.user_name:
                response = f"👋 Greetings, Space Agent! I'm {self.agent_name}, your AI mission support system.\n\nI can help with:\n🚨 Emergencies (oxygen, fire, hull breach, etc.)\n⚙️ System diagnostics\n🏥 Medical situations\n📡 Communication issues\n🧠 Psychological support\n\nWhat's your call sign (name)?"
            else:
                response = f"Welcome back, Agent {self.user_name}! All systems ready. How can I assist?"

        # Farewell
        elif re.search(r"\b(bye|goodbye|see you|exit|quit)\b", message_lower):
            response = f"🛰️ Safe travels, Agent {self.user_name if self.user_name else ''}!\n\nMission Control standing by. We're here 24/7 when you need us.\n\nStay safe among the stars! 🌟"

        # Status check
        elif re.search(
            r"\b(status|report)\b", message_lower
        ) and not self.detect_emergency_intent(message):
            response = f"📊 SYSTEM STATUS - {datetime.now().strftime('%H:%M:%S UTC')}\n\n✓ AI: ONLINE\n✓ Knowledge Base: {len(self.dataset)} protocols\n✓ Comms: NOMINAL\n✓ Mission: {self.mission_status}\n\nReady to assist!"

        # Thank you
        elif re.search(r"\b(thank|thanks)\b", message_lower):
            response = (
                "You're welcome, agent! Always here to help. What else do you need?"
            )

        # LAYER 3: Check for name introduction
        elif not self.user_name:
            name_patterns = [
                r"(?:name is|i'm|im|i am|call me)\s+(\w+)",
                r"^(\w+)\s+(?:here|reporting)",
                r"this is\s+(\w+)",
            ]
            name_found = False
            for pattern in name_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    self.user_name = match.group(1).capitalize()
                    response = f"✅ Call sign registered: Agent {self.user_name}\n\nExcellent! I'm ready to assist you with any situation.\n\nWhat do you need help with?"
                    name_found = True
                    break

            if not name_found:
                # Continue to emergency/query handling
                pass
            else:
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "message": response,
                        "timestamp": datetime.now(),
                    }
                )
                return response

        # LAYER 4: Emergency detection with fuzzy matching
        is_emergency = self.detect_emergency_intent(message)

        if is_emergency:
            # Try to find what kind of emergency
            matches = self.fuzzy_keyword_match(message)

            if matches and matches[0][1] >= 5:  # Good match found
                best_match = self.dataset[matches[0][0]]
                response = f"⚠️ EMERGENCY DETECTED - {best_match['category'].upper()}\n\n{best_match['response']}"
                self.last_topic = best_match["category"]

            else:
                # Emergency but unclear type - ask for clarification
                emergency_types = self.extract_emergency_type(message)

                if emergency_types:
                    # Found potential types
                    options = []
                    for idx, entry in enumerate(self.dataset):
                        for et in emergency_types:
                            if et in entry["category"].lower() or any(
                                et in kw for kw in entry["keywords"]
                            ):
                                if entry not in options:
                                    options.append(entry)

                    if len(options) == 1:
                        response = f"⚠️ EMERGENCY DETECTED\n\n{options[0]['response']}"
                        self.last_topic = options[0]["category"]
                    elif len(options) > 1:
                        response = "🚨 EMERGENCY DETECTED! Please specify:\n\n"
                        self.clarification_options = options
                        for i, opt in enumerate(options[:5]):
                            response += f"{i + 1}. {opt['category'].upper().replace('_', ' ')}\n"
                        response += "\nWhich emergency are you experiencing? (Type number or name)"
                        self.awaiting_clarification = True
                    else:
                        response = self.general_emergency_response()
                else:
                    response = self.general_emergency_response()

        # LAYER 5: Fuzzy keyword matching for queries
        else:
            matches = self.fuzzy_keyword_match(message)

            if matches and matches[0][1] >= 3:  # Some match found
                # Check if multiple good matches
                good_matches = [m for m in matches if m[1] >= 8]

                if len(good_matches) > 1:
                    # Multiple possibilities - ask for clarification
                    response = "I found multiple topics that might help:\n\n"
                    self.clarification_options = []
                    for i, (idx, score) in enumerate(good_matches[:5]):
                        entry = self.dataset[idx]
                        self.clarification_options.append(entry)
                        response += (
                            f"{i + 1}. {entry['category'].upper().replace('_', ' ')}\n"
                        )
                    response += "\nWhich one do you need? (Type number or name)"
                    self.awaiting_clarification = True

                else:
                    # Single best match
                    best_match = self.dataset[matches[0][0]]
                    response = best_match["response"]
                    self.last_topic = best_match["category"]

                    # Add follow-up questions if available
                    if best_match.get("questions"):
                        response += f"\n\n❓ To help further:\n"
                        for q in best_match["questions"][:3]:
                            response += f"• {q}\n"

            # LAYER 6: Contextual follow-up
            elif self.last_topic and len(self.conversation_history) > 2:
                response = (
                    f"I'm still here to help with your {self.last_topic} situation.\n\n"
                )
                response += "Could you provide more details? Or ask about:\n"
                response += "• Different emergency\n"
                response += "• System check\n"
                response += "• Space information\n"
                response += "• Psychological support"

            # LAYER 7: Ultimate fallback with helpful suggestions
            else:
                response = "🤔 I want to help but need more details.\n\n"
                response += "I'm trained on:\n\n"
                response += "🚨 EMERGENCIES:\n"
                response += "• Oxygen/air problems\n"
                response += "• Fire\n"
                response += "• Hull breach/pressure loss\n"
                response += "• Radiation exposure\n"
                response += "• Medical injuries\n\n"
                response += "⚙️ SYSTEMS:\n"
                response += "• Power/electrical\n"
                response += "• Communication\n"
                response += "• Navigation\n"
                response += "• Life support\n\n"
                response += "🌌 INFORMATION:\n"
                response += "• Planets & space facts\n"
                response += "• Mission data\n\n"
                response += "🧠 SUPPORT:\n"
                response += "• Stress & anxiety help\n\n"
                response += "What do you need help with?"

        # Store response
        self.conversation_history.append(
            {"role": "assistant", "message": response, "timestamp": datetime.now()}
        )

        return response

    def general_emergency_response(self) -> str:
        """Response when emergency detected but type unclear"""
        return """🚨 EMERGENCY PROTOCOL ACTIVATED

I detect an emergency but need to know the type. Please specify:

1. 💨 OXYGEN/AIR - Leak, low O2, can't breathe
2. 🔥 FIRE - Flames, smoke, burning
3. 🕳️ HULL BREACH - Hole, depressurization, pressure loss
4. ☢️ RADIATION - Solar flare, high dosimeter reading
5. ⚡ POWER FAILURE - Electrical issues, battery dead
6. 📡 COMMUNICATION LOSS - Can't reach Earth
7. 🏥 MEDICAL - Injury, illness, unconscious crew
8. 🧭 NAVIGATION - Lost, off course
9. 🌬️ LIFE SUPPORT - CO2 high, temperature issues

Type the NUMBER or NAME of your emergency for immediate protocol!"""

    def save_custom_dataset_template(
        self, filepath: str = "custom_dataset_template.json"
    ):
        """Save template for custom datasets"""
        template = [
            {
                "keywords": ["thruster", "rcs", "attitude", "rotation", "maneuver"],
                "response": "🚀 THRUSTER SYSTEM PROTOCOL\n\n[Your detailed response with procedures]\n\nInclude:\n• Immediate actions\n• Diagnostics\n• Troubleshooting steps\n• Safety warnings",
                "severity": "HIGH",
                "category": "propulsion",
                "questions": [
                    "Which thrusters are affected?",
                    "Can you rotate the spacecraft?",
                    "What's your fuel level?",
                ],
            },
            {
                "keywords": ["your", "keywords", "here"],
                "response": "Your detailed response",
                "severity": "CRITICAL/HIGH/MEDIUM/LOW/INFO",
                "category": "category_name",
                "questions": ["Optional", "follow-up", "questions"],
            },
        ]

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
            print(f"✅ Template saved to {filepath}")
            print("\n📝 Format:")
            print("• keywords: List of words users might say")
            print("• response: Your detailed protocol/info")
            print("• severity: CRITICAL/HIGH/MEDIUM/LOW/INFO")
            print("• category: For organization")
            print("• questions: Optional follow-up questions")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    def export_conversation(self, filepath: str):
        """Export chat history"""
        try:
            history = []
            for entry in self.conversation_history:
                history.append(
                    {
                        "role": entry["role"],
                        "message": entry["message"],
                        "timestamp": entry["timestamp"].isoformat(),
                    }
                )

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            print(f"✅ Conversation saved to {filepath}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    def chat(self):
        """Interactive chat interface"""
        print("=" * 80)
        print("🚀 SOS: SPACE AGENT - INTELLIGENT MISSION SUPPORT 🚀".center(80))
        print("=" * 80)
        print(f"\n{self.agent_name}: AI systems online...")
        print(f"{self.agent_name}: {len(self.dataset)} emergency protocols loaded")
        print(f"{self.agent_name}: Multi-layer intelligence active 🧠")
        print(f"{self.agent_name}: Mission Control link established 🛰️\n")
        print("=" * 80)
        print("\n💡 INTELLIGENCE FEATURES:")
        print("   ✓ Understands natural language (not just exact keywords)")
        print("   ✓ Detects emergencies automatically")
        print("   ✓ Asks clarifying questions when needed")
        print("   ✓ Remembers conversation context")
        print("   ✓ Fuzzy keyword matching")
        print("\n📝 COMMANDS:")
        print("   • 'export' - Save conversation")
        print("   • 'template' - Get dataset format")
        print("   • 'quit' - Exit")
        print("\n💬 TRY SAYING:")
        print('   • "Emergency! We\'re losing air!"')
        print('   • "Something\'s burning"')
        print('   • "I can\'t reach Earth"')
        print('   • "Tell me about Mars"')
        print('   • "I\'m feeling anxious"')
        print("\n" + "=" * 80 + "\n")

        while True:
            try:
                user_input = input(
                    f"Agent {self.user_name if self.user_name else '[You]'}: "
                ).strip()

                if not user_input:
                    continue

                # Special commands
                if user_input.lower() == "export":
                    filename = (
                        f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    self.export_conversation(filename)
                    continue

                if user_input.lower() == "template":
                    self.save_custom_dataset_template()
                    continue

                # Exit check
                if user_input.lower() in ["quit", "exit"]:
                    print(f"\n{self.agent_name}: {self.generate_response('bye')}")
                    break

                # Generate smart response
                response = self.generate_response(user_input)
                print(f"\n{self.agent_name}: {response}\n")

            except KeyboardInterrupt:
                print(f"\n\n{self.agent_name}: ⚠️ Emergency disconnect detected!")
                print(
                    f"{self.agent_name}: Stay safe, agent! Mission Control standing by 🚀\n"
                )
                break
            except Exception as e:
                print(f"\n{self.agent_name}: ⚠️ System error: {str(e)}")
                print(f"{self.agent_name}: Recalibrating... Try again.\n")


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================


def run_demo():
    """Run demonstration of intelligent features"""
    print("\n🧪 INTELLIGENCE DEMO - See how the bot understands natural language\n")

    bot = SmartSpaceAgentChatbot()

    test_cases = [
        "Hi, I'm Alex",
        "Emergency! We're losing oxygen fast!",
        "Help! Something is on fire!",
        "I can't reach mission control",
        "Tell me about the red planet",
        "I'm feeling really stressed out",
        "What if there's a hole in the ship?",
        "The batteries are dying",
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {test}")
        print(f"{'=' * 80}")
        response = bot.generate_response(test)
        print(f"\n{bot.agent_name}: {response[:500]}...")
        print()
        input("Press Enter for next test...")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys

    print("\n🤖 SOS: Space Agent Chatbot Initializing...\n")
    print("Choose mode:")
    print("1. Interactive Chat (default)")
    print("2. Run Intelligence Demo")

    choice = input("\nEnter choice (1 or 2, press Enter for 1): ").strip()

    if choice == "2":
        run_demo()
    else:
        # Interactive mode
        bot = SmartSpaceAgentChatbot()

        # Optional: Load custom dataset
        # bot.load_dataset("my_custom_data.json")

        bot.chat()

    print("\n" + "=" * 80)
    print("Mission Control disconnected. Safe travels! 🌟")
    print("=" * 80 + "\n")
