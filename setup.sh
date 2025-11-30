#!/bin/bash

# NextHorizon Setup Script
# This script configures the application for immediate use

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         NextHorizon - Automated Setup Script              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file from template"
        echo "⚠️  IMPORTANT: Please add your OpenAI API key to .env file"
        echo "   Edit .env and set: OPENAI_API_KEY=your-api-key-here"
    else
        echo "❌ .env.example not found. Creating basic .env..."
        cat > .env << 'EOF'
# OpenAI API Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Application Settings
APP_NAME=NextHorizon
DEBUG_MODE=false
SHOW_SIDEBAR=false
EOF
        echo "✅ Created basic .env file"
        echo "⚠️  Please add your OpenAI API key to .env file"
    fi
else
    echo "✅ .env file already exists"
fi

# Check if databases exist
echo ""
echo "📊 Checking databases..."

if [ -f "build_jd_dataset/jd_database.csv" ]; then
    echo "✅ Job description database found"
else
    echo "⚠️  Job description database not found at build_jd_dataset/jd_database.csv"
fi

if [ -f "build_training_dataset/training_database.csv" ]; then
    echo "✅ Training course database found"
else
    echo "⚠️  Training course database not found at build_training_dataset/training_database.csv"
fi

# Check Python and dependencies
echo ""
echo "🐍 Checking Python environment..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ $PYTHON_VERSION found"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing/updating Python dependencies..."
    pip3 install -q -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "⚠️  requirements.txt not found"
fi

# Install spaCy model if spaCy is present
if python3 -c "import importlib,sys
try:
    importlib.import_module('spacy')
    sys.exit(0)
except Exception:
    sys.exit(1)
" >/dev/null 2>&1; then
    echo "🧠 spaCy detected; attempting to download small English model (en_core_web_sm)"
    python3 -m spacy download en_core_web_sm || echo "⚠️ spaCy model download failed; you can run: python -m spacy download en_core_web_sm"
else
    echo "ℹ️ spaCy not installed; skipping model download. To enable NER/POS install spaCy and run: python -m spacy download en_core_web_sm"
fi

# Create necessary directories
echo ""
echo "📁 Creating necessary directories..."
mkdir -p .streamlit_tmp
mkdir -p logs
echo "✅ Directories created"

# Create Streamlit config for better UI
echo ""
echo "⚙️  Configuring Streamlit..."
mkdir -p .streamlit

cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#2c3e50"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[runner]
magicEnabled = true
fastReruns = true
EOF

echo "✅ Streamlit configured with custom theme"

# Display OpenAI API key status
echo ""
echo "🔑 Checking OpenAI API key..."
if [ -f ".env" ]; then
    if grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
        echo "✅ OpenAI API key appears to be configured"
    else
        echo "⚠️  OpenAI API key not found or invalid in .env"
        echo "   Get your API key from: https://platform.openai.com/api-keys"
        echo "   Then update .env file: OPENAI_API_KEY=sk-your-key-here"
    fi
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 To start the application, run:"
echo "   streamlit run app.py"
echo ""
echo "📖 The app will be available at: http://localhost:8501"
echo ""
echo "✨ Features configured:"
echo "   • Enhanced modern UI with gradients and animations"
echo "   • Pre-loaded databases (if available)"
echo "   • Hidden sidebar for cleaner interface"
echo "   • Custom theme (purple gradient)"
echo "   • Auto-initialization on startup"
echo ""
echo "⚠️  Remember to:"
echo "   1. Add your OpenAI API key to .env file"
echo "   2. Ensure database files are in correct locations"
echo "   3. Check that port 8501 is available"
echo ""
echo "Need help? Check README.md or PROJECT_OVERVIEW.md"
echo ""
