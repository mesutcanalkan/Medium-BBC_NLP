"""_summary_
"""
icis_config = {}
icis_config['icis_api_folder_path'] = '/Users/mesutcanalkan/energy_data'
icis_config['icis_pencepertherm_to_poundspermwh'] = '2.9307111'
icis_config['check_and_collect_icis_api_data'] = 'True'

logging_config = {}
logging_config['logging_format'] = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'
logging_config['logging_level'] = 'INFO'
logging_config['logging_filemode'] = 'w+'

selenium_config = {}
selenium_config['timeout'] = '10'
selenium_config['headless'] = 'True'
selenium_config['window_size'] = '--window-size=1920,1080'
selenium_config['num_retries'] = '3'
