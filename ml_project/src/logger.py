import logging
import os
from datetime import datetime

# Logger format
log_file_name = f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.log"
logs_directory_path = os.path.join(os.getcwd(), 'logs', log_file_name)
os.makedirs(name=logs_directory_path, exist_ok=True)
log_file_path = os.path.join(logs_directory_path, log_file_name)

# Logger config
logging.basicConfig(filename=log_file_path,
                    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s- %(message)s',
                    level=logging.INFO
                    )

