import sys
import zmq
import multiprocessing
from contextlib import nullcontext


class CallbackFactory:
    def __init__(self, callback_class, args=()):
        self.callback_class = callback_class
        self.args = args

    def __call__(self):
        return self.callback_class(*self.args)


class Response:
    def __init__(self, socket):
        self.socket = socket
        self.status_code = 200
        self.body = ''
        self.committed = False

    def set_status_code(self, code):
        self.status_code = code

    def set_body(self, body):
        self.body = body

    def commit(self):
        if self.committed is False:
            self.socket.send_pyobj((self.status_code, self.body))
            self.committed = True


def _shutdown_server(socket_address):
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.connect(socket_address)
    socket.send_pyobj('shutdown')
    socket.close()


def _server_entry(socket_address: str, callback):
    if sys.platform == 'win32':
        from zmq.utils.win32 import allow_interrupt

        def _interrupt_handler():
            _shutdown_server(socket_address)

        interrupt_context = allow_interrupt(_interrupt_handler)
    else:
        interrupt_context = nullcontext()

    if isinstance(callback, CallbackFactory):
        callback = callback()
    socket = zmq.Context.instance().socket(zmq.REP)
    socket.bind(socket_address)

    with interrupt_context:
        while True:
            command = socket.recv_pyobj()
            if command == 'hello':
                socket.send_pyobj((200,))
            elif command == 'shutdown':
                socket.close()
                return
            else:
                response = Response(socket)
                callback(command, response)
                response.commit()


class ServerLauncher:
    def __init__(self, socket_address: str, callback):
        self.socket_address = socket_address
        self.callback = callback
        self.process = None
        self.stopped = None

    @staticmethod
    def try_bind_address(socket_address):
        try:
            socket = zmq.Context.instance().socket(zmq.REP)
            socket.bind(socket_address)
            socket.close()
            del socket
            return True
        except Exception:
            return False

    def __del__(self):
        self.stop()

    def is_launched(self):
        return self.stopped is False

    def launch(self, wait_for_ready=True):
        self.process = multiprocessing.Process(target=_server_entry, args=(self.socket_address, self.callback))
        self.process.start()
        if wait_for_ready:
            socket = zmq.Context.instance().socket(zmq.REQ)
            socket.connect(self.socket_address)
            socket.send_pyobj('hello')
            recv = socket.recv_pyobj()
            socket.close()
            if recv[0] != 200:
                self.process.kill()
                raise Exception('Unexpected value')
        self.stopped = False

    def stop(self, wait_for_stop=False, waiting_timeout=5):
        if self.stopped is False:
            _shutdown_server(self.socket_address)
            if wait_for_stop:
                self.process.join(waiting_timeout)
                if self.process.exitcode is None:
                    self.process.kill()
                    print('Timeout when waiting for server process to exit. Killed.')
                self.process.close()
                del self.process
            self.stopped = True


class Client:
    def __init__(self, socket_address: str):
        self.socket_address = socket_address

    def _initialize(self):
        if not hasattr(self, 'socket'):
            self.socket = zmq.Context.instance().socket(zmq.REQ)
            self.socket.connect(self.socket_address)

    def start(self):
        self._initialize()

    def stop(self):
        if hasattr(self, 'socket'):
            self.socket.close()
            del self.socket

    def __call__(self, *args):
        self._initialize()
        self.socket.send_pyobj(args)
        response = self.socket.recv_pyobj()
        if response[0] < 200 or response[0] >= 300:
            raise RuntimeError(f'remote procedure failed with code {response[0]}')
        else:
            return response[1]
