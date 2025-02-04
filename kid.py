import aiml
import spacy
import streamlit as st

class SmartChatBot:
    def __init__(self):
        self.kernel = aiml.Kernel()
        self.kernel.learn("kid.aiml")
        self.nlp = spacy.load("en_core_web_sm")
        
        self.device_states = {
            "lights": {"status": "off", "brightness": 0},
            "fan": {"status": "off", "speed": 0},
            "ac": {"status": "off", "temperature": 24},
        }
        
        self.device_synonyms = {
            "light": "lights", "lamp": "lights", "bulb": "lights",
            "fan": "fan", "blower": "fan",
            "ac": "ac", "aircon": "ac", "air conditioner": "ac"
        }

    def handle_status_query(self, device_name):
        device_name = device_name.lower().strip()
        device = self.device_synonyms.get(device_name, device_name)
        
        if device not in self.device_states:
            similar = [d for d in self.device_states if d.startswith(device[:3])]
            if similar:
                return f"🔍 I don't see {device_name}. Did you mean {', '.join(similar)}?"
            return f"🔍 I don't recognize {device_name}. I monitor lights, fan, and AC."
        
        state = self.device_states[device]
        status = state['status'].upper()
        response = f"📟 {device.upper()} STATUS:\n- POWER: {status}"
        
        if device == "lights":
            response += f"\n- BRIGHTNESS: {state['brightness']}%"
        elif device == "fan":
            response += f"\n- SPEED: {state['speed']}%"
        elif device == "ac":
            response += f"\n- TEMPERATURE: {state['temperature']}°C"
            
        return response

    def preprocess_input(self, user_input):
        user_input = user_input.strip().lower()
        doc = self.nlp(user_input)
        
        # Detect intents
        intents = {
            "capabilities": ["can you do", "capabilities", "features"],
            "self_description": ["who are you", "what are you", "tell me about yourself"],
            "status_query": ["status", "is the", "are the", "check", "what's"],
            "device_control": ["turn", "set", "adjust", "change", "switch"]
        }
        
        for intent, keywords in intents.items():
            if any(kw in user_input for kw in keywords):
                return [{"intent": intent, "text": user_input}]
        
        return [{"intent": "general", "text": user_input}]

    def handle_device_command(self, command):
        doc = self.nlp(command.lower())
        action = None
        device = None
        value = None

        # Detect action
        action_map = {
            "on": ["on", "activate", "enable"],
            "off": ["off", "deactivate", "disable"],
            "set": ["set", "adjust", "change"]
        }
        
        for token in doc:
            for action_type, keywords in action_map.items():
                if token.text in keywords:
                    action = action_type
                    break

        # Detect device
        for token in doc:
            if token.text in self.device_synonyms:
                device = self.device_synonyms[token.text]
                break
            if token.text in self.device_states:
                device = token.text
                break

        # Detect value
        for ent in doc.ents:
            if ent.label_ in ["PERCENT", "CARDINAL"]:
                value = int(ent.text.replace('%', ''))

        # Handle missing device
        if not device:
            return "❓ Which device would you like to control? I manage: lights, fan, AC"

        # Handle unknown device
        if device not in self.device_states:
            return f"⚠️ I can't control {device}. Available devices: lights, fan, AC."

        # Execute command
        try:
            if action == "on":
                self.device_states[device]["status"] = "on"
                if device == "lights": 
                    self.device_states[device]["brightness"] = 100
                elif device == "fan":
                    self.device_states[device]["speed"] = 50
                return f"✅ {device.capitalize()} POWER ON"
                
            elif action == "off":
                self.device_states[device]["status"] = "off"
                if device == "lights": 
                    self.device_states[device]["brightness"] = 0
                elif device == "fan":
                    self.device_states[device]["speed"] = 0
                return f"✅ {device.capitalize()} POWER OFF"
                
            elif action == "set" and value is not None:
                if device == "lights":
                    if 0 <= value <= 100:
                        self.device_states[device]["brightness"] = value
                        return f"🔆 LIGHT BRIGHTNESS SET TO {value}%"
                    return "⚠️ Please enter value between 0-100%"
                elif device == "fan":
                    if 0 <= value <= 100:
                        self.device_states[device]["speed"] = value
                        return f"🌀 FAN SPEED SET TO {value}%"
                    return "⚠️ Please enter value between 0-100%"
                elif device == "ac":
                    if 16 <= value <= 30:
                        self.device_states[device]["temperature"] = value
                        return f"❄️ AC TEMPERATURE SET TO {value}°C"
                    return "⚠️ Please enter temperature between 16-30°C"

        except Exception as e:
            return f"⚠️ Error: {str(e)}"

        return "⚠️ Couldn't process command. Try: 'Set lights to 50%' or 'Turn off fan'"

    def handle_command(self, user_input):
        processed = self.preprocess_input(user_input)[0]
        
        if processed["intent"] == "capabilities":
            return "I can:\n- Control devices (lights/fan/AC)\n- Answer questions\n- Check device statuses\n- Have conversations!"
        
        if processed["intent"] == "self_description":
            return "I'm KID: Knowledge Interface Device\nAI-powered Smart Home Assistant"
        
        if processed["intent"] == "status_query":
            doc = self.nlp(processed["text"])
            device = None
            
            # Extract device using NER
            for ent in doc.ents:
                if ent.text.lower() in self.device_synonyms:
                    device = self.device_synonyms[ent.text.lower()]
                    break
                    
            if not device:
                for token in doc:
                    if token.text in self.device_synonyms:
                        device = self.device_synonyms[token.text]
                        break
                    elif token.text in self.device_states:
                        device = token.text
                        break
                        
            return self.handle_status_query(device) if device else \
                "Which device status would you like to check? (lights/fan/ac)"
        
        if processed["intent"] == "device_control":
            return self.handle_device_command(processed["text"])
        
        # Fallback to AIML
        aiml_response = self.kernel.respond(user_input.upper())
        return aiml_response if aiml_response else "🤔 I'm still learning! How else can I assist you?"

def main():
    st.set_page_config(page_title="Smart Chatbot", page_icon="🤖")
    st.title("🏠 Smart Home Assistant KID")
    st.write("Control devices | Check statuses | Ask questions")
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = SmartChatBot()
    
    if "history" not in st.session_state:
        st.session_state.history = []
        
    # Display chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # User input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        response = st.session_state.chatbot.handle_command(prompt)
        
        # Add bot response
        st.session_state.history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()