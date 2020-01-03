import ctypes
from typing import Awaitable

import websockets
import asyncio
import numpy as np

from .constants import BIT_MASK_HAND_LEFT, BIT_MASK_HAND_RIGHT
from .hand_detection import CascadeClassifierHandDetector

__all__ = [
    'WebSocketServer',
    'HandBEATServer'
]


class WebSocketServer:
    _routes = dict()

    def __init__(self, host='127.0.0.1', port=8080, max_size=6220800, logging: bool = False):
        self._ws: websockets.server.Serve = websockets.serve(self.handle_connection, host, port, max_size=max_size)
        self._loop = asyncio.get_event_loop()
        self._loop.run_until_complete(self._ws)
        self.logging = logging

    def start(self):
        if self.logging:
            print('Starting server')
        self._loop.run_forever()

    def stop(self):
        if self.logging:
            print('Stopping server')
        # TODO: Graceful shutdown
        self._loop.stop()

    @staticmethod
    def split_path(path: str):
        return path.lstrip('/').rstrip('/').split('/')

    @classmethod
    def register_route(cls, path):
        def set_callback(f):
            path_parts = WebSocketServer.split_path(path)

            route_node = cls._routes
            for part in path_parts[:-1]:
                if part not in route_node:
                    route_node[part] = dict()

                route_node = route_node[part]

            route_node[path_parts[-1]] = f

            return f

        return set_callback

    @asyncio.coroutine
    def handle_connection(self, ws: websockets.WebSocketServerProtocol, path: str) -> Awaitable:
        if self.logging:
            print('[{}:{}]: Client connecting to {}'.format(
                ws.remote_address[0],
                ws.remote_address[1],
                path
            ))
        parts = WebSocketServer.split_path(path)

        route_node = self._routes
        part_index = None
        for index, part in enumerate(parts):
            if part in route_node:
                route_node = route_node[part]
                part_index = index

                if type(route_node) is not dict:
                    break
            else:
                if self.logging:
                    print('[{}:{}]: rejected (invalid path)'.format(
                        ws.remote_address[0],
                        ws.remote_address[1],
                        path
                    ))
                return

        if callable(route_node) and part_index is not None:
            if self.logging:
                print('[{}:{}]: connected'.format(
                    ws.remote_address[0],
                    ws.remote_address[1],
                    path
                ))
            yield from route_node(self, ws, parts[part_index + 1:])


class HandBEATServer(WebSocketServer):
    @WebSocketServer.register_route('/detect/hands')
    @asyncio.coroutine
    def handle_detect_hands(self, ws: websockets.WebSocketServerProtocol, path_parts: list):
        if len(path_parts) > 0:
            resolution_parts = path_parts[0].split('x')
            if len(resolution_parts) == 2:
                shape = tuple(map(int, list(reversed(resolution_parts)) + [-1]))
                if self.logging:
                    print('[{}:{}, detect_hands]: set resolution to {}x{}'.format(
                        ws.remote_address[0],
                        ws.remote_address[1],
                        resolution_parts[0],
                        resolution_parts[1]
                    ))
                hand_detector = CascadeClassifierHandDetector()
                while True:
                    buffer = yield from ws.recv()
                    image = np.frombuffer(buffer, dtype=np.uint8).reshape(shape)
                    image = image[:, :, :3][:, :, ::-1]
                    hand_left, hand_right = hand_detector.detect(image)

                    buffer = b''

                    hands_mask = 0x0
                    if hand_left is not None:
                        hands_mask |= BIT_MASK_HAND_LEFT
                    else:
                        hand_left = np.zeros((2,), np.float)
                    if hand_right is not None:
                        hands_mask |= BIT_MASK_HAND_RIGHT
                    else:
                        hand_right = np.zeros((2,), np.float)

                    buffer += bytes(ctypes.c_uint8(hands_mask))

                    # hand_left, as well as hand_right, should have the shape (2,),
                    # resulting in a combined shape of (2, 2),
                    # which occupies 2 x 2 x 4 = 16 bytes of memory
                    buffer += np.stack([
                        hand_left,
                        hand_right
                    ]).astype(np.float32).data

                    yield from ws.send(buffer)
