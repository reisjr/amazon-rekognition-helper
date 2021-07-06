BLUE='\033[0;34m'
NC='\033[0m' # No Color

init:
	virtualenv -p python3 .env
	source .env/bin/activate && pip install -r requirements.txt