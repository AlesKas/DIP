
# Prerequisites
Before you begin, ensure you have met the following requirements:
- You have installed Python 3.11.9 on your machine.

## Setting Up Python Environment
To set up the Python environment and install the required Python packages, follow these steps:

1. Create a virtual environment (optional, but recommended):

   ```bash
    python3 -m venv venv
    source venv/bin/activate
   ```

2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Additional Dependencies
    Some functionalities in this project require Graphviz. To install Graphviz on Debian-based systems, run the following command:

    ```bash
    sudo apt-get install graphviz
    ```