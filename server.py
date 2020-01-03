import argparse
import atexit

from hand_beat import HandBEATServer


def get_cleanup_callback(server):
    def cleanup():
        server.stop()

    return cleanup


def main():
    parser = argparse.ArgumentParser(description='HandBEAT Backend')
    parser.add_argument('--host', type=str, dest='host', default='127.0.0.1')
    parser.add_argument('--port', type=int, dest='port', default=8080)

    args = parser.parse_args()

    server = HandBEATServer(host=args.host, port=args.port, logging=True)

    atexit.register(get_cleanup_callback(server))

    server.start()


if __name__ == '__main__':
    main()
