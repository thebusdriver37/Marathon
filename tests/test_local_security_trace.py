"""Allow only traced loopback listeners in the offline startup audit."""
import unittest
from scripts.test_local_security import local_listener_ports


def event(pid, operation, fd, port, address='127.0.0.1', result='0'):
    return f'{pid} {operation}({fd}, {{sa_family=AF_INET, sin_port=htons({port}), sin_addr=inet_addr("{address}")}}, [16]) = {result}'


class LocalListenerTests(unittest.TestCase):
    def test_ephemeral_listener_requires_matching_bind(self):
        trace = '\n'.join([event(10, 'bind', 7, 0), event(10, 'getsockname', 7, 35729)])
        self.assertEqual(local_listener_ports(trace), {'35729'})

    def test_outgoing_socket_is_not_a_listener(self):
        self.assertEqual(local_listener_ports(event(10, 'getsockname', 7, 35729)), set())

    def test_public_and_failed_binds_are_not_allowed(self):
        trace = '\n'.join([event(10, 'bind', 7, 35729, '0.0.0.0'),
                           event(10, 'bind', 8, 35730, result='-1 EACCES')])
        self.assertEqual(local_listener_ports(trace), set())

    def test_other_process_or_fd_cannot_claim_listener(self):
        trace = '\n'.join([event(10, 'bind', 7, 0), event(11, 'getsockname', 7, 35729),
                           event(10, 'getsockname', 8, 35730)])
        self.assertEqual(local_listener_ports(trace), set())
