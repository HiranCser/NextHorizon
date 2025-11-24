# 🧭 NextHorizon - Enhanced UI Version

## 🎨 What's New in This Version

### Visual Enhancements
- ✨ **Modern Gradient Design**: Beautiful purple gradient theme throughout
- 🎯 **Hero Sections**: Each tab has an attractive hero section explaining its purpose
- 📊 **Interactive Cards**: Enhanced card-based layouts with hover effects
- 🎭 **Color-Coded Elements**: Match scores, skill gaps, and progress indicators with intuitive colors
- 📈 **Progress Indicators**: Visual representations of role readiness and skill matching
- 💫 **Smooth Animations**: Hover effects and transitions for better UX

### User Experience Improvements
- 🚀 **One-Click Setup**: Run `./setup.sh` for complete configuration
- 📦 **Pre-loaded Databases**: Databases load automatically on startup
- 🎛️ **Hidden Sidebar**: Cleaner interface with no sidebar clutter
- 📱 **Responsive Design**: Works seamlessly on different screen sizes
- ⚡ **Faster Navigation**: Streamlined workflow across all tabs
- 💡 **Better Guidance**: Clear instructions and status indicators

### Functional Enhancements
- ✅ **Auto-Initialization**: All databases and settings load on app start
- 🔄 **Status Dashboard**: Real-time database and API status indicators
- 🎯 **Enhanced Matching**: Better visual representation of role matches
- 📊 **Skill Analytics**: Detailed breakdown with match percentages
- 📚 **Course Cards**: Beautiful course presentation with direct enrollment links
- 🔍 **Improved Search**: Better course filtering and recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository** (if not already done)
   ```bash
   cd ~/NextHorizon
   ```

2. **Run the automated setup**
   ```bash
   ./setup.sh
   ```

3. **Add your OpenAI API key**
   Edit the `.env` file and add your key:
   ```bash
   nano .env
   # Set: OPENAI_API_KEY=sk-your-key-here
   ```

4. **Start the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   Open your browser to: `http://localhost:8501`

## 🎯 How to Use

### Step 1: Resume Analysis 📄
- Upload your resume (PDF, DOCX, or TXT)
- AI extracts and structures your information
- Review and edit the parsed data
- Add or remove work experience and education entries

### Step 2: Role Matching 🎯
- Enter your career aspirations
- AI analyzes and matches you with relevant roles
- See percentage match for each role
- Explore specific job openings

### Step 3: Skill Gap Analysis 🔍
- View your current skills
- Identify skills you need to develop
- See your role readiness percentage
- Interactive skill breakdown with color coding

### Step 4: Learning Path 📚
- Get personalized course recommendations
- Courses matched to your skill gaps
- Direct enrollment links
- Courses organized by skill category

## 🎨 UI Features

### Design Elements
- **Gradient Headers**: Purple to violet gradient for section headers
- **Status Badges**: Green (good), orange (needs attention), blue (info)
- **Match Indicators**: Color-coded percentage badges
- **Skill Cards**: White cards with colored left borders
- **Course Cards**: Detailed course information with provider and duration

### Color Scheme
- **Primary**: `#667eea` (Purple)
- **Secondary**: `#764ba2` (Violet)
- **Success**: `#4caf50` (Green)
- **Warning**: `#ff9800` (Orange)
- **Info**: `#2196f3` (Blue)
- **Background**: `#f8f9fa` (Light Gray)

## 📊 Database Structure

### Job Descriptions Database
**Location**: `build_jd_dataset/jd_database.csv`

Required columns:
- `role_title`: Job role name
- `company`: Company name
- `jd_text`: Job description text
- `source_title`: Job posting title
- `source_url`: Link to job posting

### Training Courses Database
**Location**: `build_training_dataset/training_database.csv`

Required columns:
- `skill`: Skill category
- `title`: Course title
- `description`: Course description
- `provider`: Course provider (Udemy, Coursera, etc.)
- `link`: Enrollment URL
- `hours`: Course duration (optional)

## 🔧 Configuration

### Environment Variables (.env)
```bash
# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Application Settings
APP_NAME=NextHorizon
DEBUG_MODE=false
SHOW_SIDEBAR=false
```

### Streamlit Configuration (.streamlit/config.toml)
Automatically created by `setup.sh` with:
- Custom purple theme
- Hidden sidebar
- Performance optimizations
- Browser settings

## 🎭 Customization

### Changing Colors
Edit `app.py` function `apply_custom_css()` to modify:
- Gradient colors
- Button styles
- Card backgrounds
- Text colors

### Adjusting Layout
Modify individual UI files:
- `ui/resume_parsing.py` - Resume upload and parsing
- `ui/role_recommendations.py` - Role matching
- `ui/skill_gaps.py` - Skill analysis
- `ui/course_recommendations.py` - Course suggestions

## 📈 Performance Tips

1. **Database Loading**: Databases are pre-loaded on startup for faster access
2. **Caching**: AI responses are processed efficiently
3. **Vector Search**: Uses OpenAI embeddings for semantic matching
4. **Streamlit Optimization**: Fast reruns enabled for better responsiveness

## 🐛 Troubleshooting

### Database Not Loading
- Check file paths in `.env`
- Ensure CSV files have correct column names
- Verify files are not empty

### API Errors
- Verify OpenAI API key is correct
- Check API usage limits
- Ensure internet connectivity

### UI Issues
- Clear browser cache
- Restart Streamlit server
- Check console for errors

### Slow Performance
- Reduce number of courses in database
- Adjust `top_k` values in slider
- Use smaller resume files

## 🔒 Security Notes

- ✅ Sidebar hidden by default (no sensitive settings exposed)
- ✅ API keys loaded from `.env` (not hardcoded)
- ✅ XSRF protection enabled
- ✅ No telemetry or usage tracking
- ⚠️ Keep `.env` file secure and never commit it to git

## 📱 Browser Compatibility

Tested and optimized for:
- ✅ Chrome/Chromium (Recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge

## 🎯 Features Roadmap

Completed:
- ✅ Enhanced modern UI
- ✅ Pre-loaded databases
- ✅ Hidden sidebar
- ✅ Auto-initialization
- ✅ Status indicators
- ✅ Improved course cards
- ✅ Better visual hierarchy

Future Enhancements:
- 🔜 Azure Cosmos DB integration
- 🔜 User authentication
- 🔜 Progress tracking
- 🔜 Chat-based career counseling
- 🔜 Real-time job data
- 🔜 Course completion tracking

## 💡 Tips for Best Results

1. **Resume Quality**: Use detailed resumes with clear skill mentions
2. **Career Aspirations**: Be specific about your goals and interests
3. **Skill Clarification**: Complete Q&A for more accurate gap analysis
4. **Database Updates**: Keep job and course databases up to date
5. **API Credits**: Monitor OpenAI usage for cost optimization

## 📞 Support

For issues or questions:
1. Check `PROJECT_OVERVIEW.md` for architecture details
2. Review `CODE_REVIEW_COMPLETE.md` for technical insights
3. Examine console logs for error messages
4. Verify all prerequisites are installed

## 🙏 Acknowledgments

Built with:
- **Streamlit**: Web framework
- **OpenAI**: GPT-4o-mini and text-embedding-3-small
- **Pandas**: Data manipulation
- **Python-docx, PyPDF2**: Document parsing

## 📄 License

See LICENSE file for details.

---

**Version**: 2.0 (Enhanced UI)  
**Last Updated**: November 2025  
**Status**: Production Ready ✅
