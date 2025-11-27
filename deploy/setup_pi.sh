#!/bin/bash
# Setup script for SushiVoice on Raspberry Pi

echo "🍣 SushiVoice Raspberry Pi Setup"
echo "================================"

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python dependencies
echo "Installing Python 3 and pip..."
sudo apt-get install -y python3 python3-pip python3-venv

# Install audio dependencies
echo "Installing audio dependencies..."
sudo apt-get install -y portaudio19-dev python3-pyaudio libsndfile1

# Install USB printer support
echo "Installing printer dependencies..."
sudo apt-get install -y libusb-1.0-0-dev

# Create virtual environment
echo "Creating Python virtual environment..."
cd /home/pi/SushiVoice
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch for ARM (CPU-only, lighter weight)
echo "Installing PyTorch for ARM..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Configure audio permissions
echo "Configuring audio permissions..."
sudo usermod -a -G audio pi

# Copy and enable systemd service
echo "Setting up systemd service..."
sudo cp deploy/sushivoice.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sushivoice.service

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy your trained model to /home/pi/SushiVoice/models/sushi_slm_quantized.pth"
echo "2. Connect USB microphone and printer"
echo "3. Test the service: sudo systemctl start sushivoice.service"
echo "4. View logs: journalctl -u sushivoice.service -f"
echo "5. Enable auto-start on boot (already done)"
echo ""
echo "To manually run (for testing):"
echo "  cd /home/pi/SushiVoice"
echo "  source venv/bin/activate"
echo "  python3 src/inference/asr_pipeline.py --model models/sushi_slm_quantized.pth --continuous"
