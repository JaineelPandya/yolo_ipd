# 🔧 Gemini Scene Descriptor - Complete Setup Guide

## ✅ What Was Fixed

Your Gemini API integration has been enhanced with:

1. **Faster Rate Limiting**: Reduced from 10 → 2 seconds between API calls
2. **Better Error Handling**: More informative logging when API fails
3. **API Key Loading**: Now properly loads from `.env` file
4. **Enhanced UI**: Query Objects tab shows Gemini status
5. **Improved Logging**: Better visibility into what's happening

---

## 📋 Current Status

| Component | Status | Details |
|-----------|--------|---------|
| API Key | ✅ Set | `AIzaSyAoE9gpy6QB2__OqNqfiXTeu6qFQ7_idjc` in `.env` |
| Python | ✅ OK | 3.13.7 (supports google-genai) |
| Model | ✅ Downloaded | `models/yolov8n.pt` (6.2 MB) |
| Database | ✅ Ready | SQLite at `data/database/object_memory.db` |

---

## 🚀 How to Use

### Step 1: Start the App
```powershell
streamlit run app_streamlit.py
```

### Step 2: Enable Gemini (Left Sidebar)
- Check: **"Enable Scene Description (Requires API Key)"**
- Status should show: ✅ API Key Set

### Step 3: Start Live Detection
1. Go to **"🎥 Live Detection"** tab
2. Click **"📹 Start Camera"**
3. Point your webcam at objects
4. YOLOv8 will detect, Gemini will describe

### Step 4: Search for Objects
1. Go to **"🔍 Query Objects"** tab
2. Enter object name (e.g., "remote", "phone", "keys")
3. Click **"🔍 Search"**
4. Get response with location description

**Example Output:**
```
🕐 Your **remote** was last seen at **03:48 PM** on Sunday, March 29.

📍 You left your remote on the wooden shelf in the bedroom, 
near the lamp on the nightstand.

✅ High confidence detection (86%)
```

---

## 🔍 How It Works

### Detection → Storage → Query Flow

```
┌─────────────────────────────────────────────────┐
│ 1. LIVE DETECTION (Camera Feed)                 │
│    - YOLOv8n detects objects (640x640)         │
│    - ByteTrack tracks across frames             │
│    - Frame saved to data/frames/                │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ 2. GEMINI SCENE DESCRIPTION (If Enabled)       │
│    - Crop frame around detected object          │
│    - Send to Gemini Vision API                  │
│    - Get: "You left your X on..."              │
│    - Cache for 2 seconds (rate limit)          │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ 3. STORE IN DATABASE                            │
│    - objects table: name, bbox, confidence      │
│    - scene_description field: Gemini output     │
│    - timestamp: when detected                   │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ 4. QUERY INTERFACE                              │
│    - User: "Where is my remote?"               │
│    - Query Engine searches database             │
│    - Retrieves last sighting + description      │
│    - Returns formatted response                 │
└─────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration Details

**File:** `config.py`

```python
# Rate limiting (seconds between API calls)
GEMINI_MIN_INTERVAL_SECONDS = 2  # Fast! One call every 2 seconds

# Prompt sent to Gemini
GEMINI_OBJECT_PROMPT_TEMPLATE = """
Look at this image. There is a '{object_name}' visible.

In 1-2 sentences, describe WHERE the {object_name} is located...
"""

# Query settings
RECENT_SEARCH_RANGE = 60  # Look back 60 minutes by default
```

---

## 🐛 Troubleshooting

### ❌ "Scene description not available"

**Causes & Solutions:**

| Issue | Solution |
|-------|----------|
| Gemini not enabled | Check sidebar: "Enable Scene Description" |
| API key not set | Check: `GEMINI_API_KEY=...` in `.env` |
| No scene_description in DB | Run live detection for a few minutes |
| Gemini API quota exceeded | Wait an hour, API has daily limits |

### ❌ Gemini API Errors

**Check logs:**
```powershell
# Run with debug logging
streamlit run app_streamlit.py --logger.level=debug
```

**Look for:**
- `✅ Gemini Scene [remote]:` = Success
- `❌ Gemini API error:` = API failed
- `⚠️ Gemini client not initialized:` = API key issue

### ✅ Debug Checklist

1. **Does example work?** Yes → your setup is complete
2. **Gemini enabled in sidebar?** Check ✅
3. **API key in .env?** Check: `GEMINI_API_KEY=AIzaSy...`
4. **Python 3.9+?** Check: `python --version`
5. **google-genai installed?** Check: `pip list | grep genai`

---

## 📊 Database Schema

**Objects Table:**
```
id, object_name, class_id, track_id, timestamp,
bbox, confidence, scene_description, image_path, ...
```

**Example Record:**
```sql
SELECT * FROM objects WHERE object_name = 'remote' LIMIT 1;

id             : 42
object_name    : remote
timestamp      : 2026-03-29 15:48:23
confidence     : 0.86
scene_desc     : "You left your remote on the wooden shelf..."
image_path     : data/frames/full/2026-03-29_154823_0.jpg
```

---

## 🎯 Tips for Best Results

### ✅ DO:
- Enable Gemini **before** starting live detection
- Keep objects **visible and clear** in frame
- Stay in **consistent lighting** (well-lit rooms work best)
- Let Gemini run for **2+ seconds** between detections (rate limit)
- Check **storage space** - frames take ~100KB each

### ❌ DON'T:
- Don't disable Gemini mid-detection (descriptions stop)
- Don't use when API key is invalid (silent fail)
- Don't expect descriptions if object is **occluded/hidden**
- Don't move objects too quickly (tracking breaks)

---

## 📈 Example Workflow

### Scenario: "Where is my remote?"

**Step 1: Live Detection (5 minutes)**
```
🎥 Camera started
  Frame 001: remote detected (0.91 confidence)
  ✅ Gemini: "remote on TV stand near the couch"
  
  Frame 045: remote detected (0.88 confidence)  
  (Skipped - 2 sec limit not reached)
  
  Frame 090: remote detected (0.92 confidence)
  ✅ Gemini: "remote on wooden shelf next to lamp"
```

**Step 2: Search (Query Objects tab)**
```
User enters: "remote"
Searches database for last sighting...
Found: 2 records

Latest record:
  - Timestamp: 03:48 PM
  - Confidence: 92%
  - Scene: "remote on wooden shelf next to lamp"
  
Response shown to user:
  "🕐 Your remote was last seen at 03:48 PM
   📍 You left your remote on the wooden shelf next to lamp
   ✅ High confidence detection (92%)"
```

---

## 🔑 API Key Management

**Your API Key:**
```
AIzaSyAoE9gpy6QB2__OqNqfiXTeu6qFQ7_idjc
```

**Stored in:** `.env

**Loaded from:** `config.py` at startup

**Rate Limit:** ~1500 requests/day (free tier)

**Pricing:** 
- First 50 requests/day = Free
- After that = $0.075 per 1K input tokens
- Images are ~200-500 tokens depending on size

---

## 📚 Next Steps

1. **Run the app:** `streamlit run app_streamlit.py`
2. **Enable Gemini** in sidebar
3. **Start camera** and point at objects for 2+ minutes
4. **Search** in Query Objects tab
5. **Check logs** if issues occur

---

## 💾 Sample Database Query

```python
# SQLite - find all objects with scene descriptions
import sqlite3
conn = sqlite3.connect('data/database/object_memory.db')
cursor = conn.cursor()

cursor.execute('''
    SELECT object_name, timestamp, scene_description, confidence
    FROM objects
    WHERE scene_description IS NOT NULL
    ORDER BY timestamp DESC
    LIMIT 10
''')

for row in cursor.fetchall():
    print(f"{row[0]} at {row[1]}: {row[2]} (confidence: {row[3]:.0%})")
```

---

## ❓ FAQ

**Q: Why does description say nothing?**
A: Gemini API is rate-limited to 1 call every 2 seconds. Wait between detections.

**Q: Can I change the description prompt?**
A: Yes! Edit `GEMINI_OBJECT_PROMPT_TEMPLATE` in `config.py`.

**Q: Does this work offline?**
A: No, Gemini requires internet and API key.

**Q: How much data does it use?**
A: Each description is ~200-500 API tokens (~10KB of text).

**Q: Can I use a different Gemini model?**
A: Yes, change `GEMINI_MODEL` in config.py. Try "gemini-pro" for more detailed responses.

---

## ✨ You're All Set!

Your system is now configured to:
- ✅ Detect objects in real-time (YOLOv8n)
- ✅ Remember where they were (SQLite)
- ✅ Describe the location (Gemini Vision API)
- ✅ Search for them naturally (Query Engine)

**Next:** Run the app and start tracking your objects!

```powershell
streamlit run app_streamlit.py
```
