# 🧠 Lowkey Genius - Your Fun Learning Companion

A voice-enabled AI teacher chatbot that makes learning engaging and interactive using natural conversation and speech synthesis.

## ✨ Features

- **🎤 Voice Input** - Speak to ask questions with speech recognition
- **🔊 Voice Output** - Hear responses in natural UK English voices (Moira or Daniel)
- **💬 Natural Conversations** - Responds like a real teacher with fun, engaging personality
- **⏸️ Pause/Resume** - Control voice playback with easy buttons
- **📚 Course Materials** - Retrieves relevant information from course notes
- **🎯 Smart Filtering** - Removes emojis and citations from speech output
- **💾 Session History** - Maintains conversation history for better context

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key
- Modern web browser (Chrome, Safari, Firefox)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/madhumini-gunaratne/teacher-chatbot.git
   cd teacher-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```bash
   touch .env
   ```
   
   Add the following variables:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```
   
   **How to get your OpenAI API key:**
   1. Visit https://platform.openai.com/api-keys
   2. Create a new API key
   3. Copy and paste it into `.env`
   
   ⚠️ **Important:** Never commit `.env` to git! It's already in `.gitignore`

5. **Run the application**
   ```bash
   python3 app.py
   ```

6. **Open in browser**
   Navigate to `http://127.0.0.1:8000`

## 📖 Usage

### Text Input
1. Type your question in the input field
2. Press "Send" or hit Enter
3. The teacher responds with an explanation

### Voice Input
1. Click the **microphone button** 🎤
2. Speak your question clearly
3. The chatbot automatically processes and responds

### Voice Controls
- **Enable Voice Output** - Toggle speech synthesis on/off
- **Voice Selection** - Choose between UK Female (Moira) or UK Male (Daniel)
- **Pause** - Pause the current speech
- **Resume** - Continue playing paused speech

## 🏗️ Project Structure

```
teacher-chatbot/
├── app.py                 # Flask backend & API routes
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not committed)
├── .gitignore            # Git ignore rules
├── README.md             # This file
│
├── templates/
│   └── index.html        # Frontend UI & JavaScript
│
├── static/
│   └── style.css         # Styling
│
└── data/
    ├── system_prompt.txt # Teacher personality prompt
    ├── index.json        # Course materials & embeddings
    └── media/
        ├── transcripts/  # Video transcripts (if any)
        └── videos/       # Video files (if any)
```

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **OpenAI API** - LLM for responses
- **Python 3.9** - Runtime

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling
- **Vanilla JavaScript** - No framework
- **Web Speech API** - Voice input/output
- **Marked.js** - Markdown rendering
- **Highlight.js** - Code syntax highlighting

## 🎓 Teaching Personality

The chatbot acts as a fun, enthusiastic teacher with:
- Natural conversational responses
- Engaging explanations with analogies
- Emoji use for personality
- Encouraging feedback
- Step-by-step breakdowns

**System Prompt:** See `data/system_prompt.txt` to customize the teaching style.

## 📊 API Endpoints

### `GET /`
Returns the main chat interface.

### `POST /chat`
Send a message to the chatbot.

**Request:**
```json
{
  "message": "What is encryption?"
}
```

**Response:**
```json
{
  "reply": "Encryption is like... [explanation]"
}
```

## 🔧 Configuration

### Models
- **Chat Model:** `gpt-3.5-turbo` (configurable in `.env`)
- **Embedding Model:** `text-embedding-3-small`

### Speech Settings
- **Speech Rate:** 0.9x (slightly slower for clarity)
- **Pitch:** 1.1 (slightly higher for clarity)
- **Voices:** UK English (Moira/Daniel)

### Content Filtering
The chatbot automatically:
- Removes emojis from speech output
- Filters parenthetical content (citations)
- Cleans markdown formatting
- Removes code blocks from voice playback

## 📝 Adding Course Materials

1. Create a JSON file in `data/source/`
2. Add your course notes or PDFs
3. Run the ingestion script:
   ```bash
   python scripts/ingest.py
   ```
4. This updates `data/index.json` with embeddings

## 🐛 Troubleshooting

### "No OPENAI_API_KEY found"
- Ensure `.env` file exists in the root directory
- Add your OpenAI API key to `.env`

### Microphone not working
- Check browser permissions for microphone access
- Ensure you're using HTTPS or localhost
- Try refreshing the page

### Voice output not playing
- Enable "Voice Output" toggle
- Check browser volume settings
- Select a voice from the dropdown

### Slow responses
- Check your internet connection
- Verify OpenAI API status
- API might be rate limiting

## 👨‍💼 Author

Created by Madhumini Gunaratne

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Happy learning with Lowkey Genius!** 🚀✨
