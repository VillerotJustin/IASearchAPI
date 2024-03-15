import api as api
import uvicorn
import sys
import traceback

# Configuration File
import config as cfg

if __name__ == '__main__':
    print(f'Starting API Server: {cfg.api["host"]}:{cfg.api["port"]}\n')

    try:
        uvicorn.run(
            "api:app",
            host=cfg.api["host"],
            port=cfg.api["port"],
            workers=cfg.api["workers"],
            log_level=cfg.api["log_level"],
            reload=cfg.api["reload"]
        )
    except KeyboardInterrupt:
        print(f'\nExiting\n')
    except Exception as e:
        print(f'Failed to Start API')
        print('=' * 100)
        traceback.print_exc(file=sys.stdout)
        print('=' * 100)
        print('Exiting\n')
    print(f'\n\n')