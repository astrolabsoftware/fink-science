# Copyright 2026 AstroLab Software
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Retry strategy shared by the science modules querying a web service

Connection errors and the server errors listed by the caller are retried twice,
at once then after 2 s. A read timeout is not retried: the request may already
have taken its whole read timeout, which the caller sets on each request.
A Retry-After header is honoured up to RETRY_AFTER_MAX seconds.
"""

import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_LOG = logging.getLogger(__name__)

# Longest wait, in seconds, granted to a Retry-After header
RETRY_AFTER_MAX = 120


class CappedRetry(Retry):
    """Retry honouring the Retry-After header of the server, up to a ceiling

    A server under maintenance may ask for a wait of an hour, during which the
    Spark task would hold its batch. The wait is logged as a warning, and
    capped to RETRY_AFTER_MAX seconds.

    Examples
    --------
    >>> from unittest import mock
    >>> from urllib3 import HTTPResponse
    >>> retry = CappedRetry(total=2)

    A long wait asked by the server is capped, and reported

    >>> busy = HTTPResponse(status=503, headers={"Retry-After": "3600"})
    >>> with mock.patch("time.sleep") as sleep, mock.patch.object(_LOG, "warning") as warn:
    ...     retry.sleep_for_retry(busy)
    True
    >>> sleep.call_args.args, warn.called
    ((120,), True)

    A short wait is honoured as is, and reported

    >>> soon = HTTPResponse(status=503, headers={"Retry-After": "5"})
    >>> with mock.patch("time.sleep") as sleep, mock.patch.object(_LOG, "warning") as warn:
    ...     retry.sleep_for_retry(soon)
    True
    >>> sleep.call_args.args, warn.called
    ((5,), True)

    Without the header, nothing is reported: the usual backoff applies

    >>> plain = HTTPResponse(status=503)
    >>> with mock.patch("time.sleep") as sleep, mock.patch.object(_LOG, "warning") as warn:
    ...     retry.sleep_for_retry(plain)
    False
    >>> sleep.called, warn.called
    (False, False)
    """

    def sleep_for_retry(self, response):
        """Wait as the server asked, at most RETRY_AFTER_MAX seconds"""
        retry_after = self.get_retry_after(response)
        if not retry_after:
            return False
        wait = min(retry_after, RETRY_AFTER_MAX)
        _LOG.warning(
            "%s asked to retry after %s s (HTTP %s), waiting %s s",
            getattr(response, "url", None) or "server",
            retry_after,
            response.status,
            wait,
        )
        time.sleep(wait)
        return True


def resilient_session(status_forcelist):
    """Session to a web service, resilient to short network outages

    Parameters
    ----------
    status_forcelist: list of int
        HTTP statuses of the answers to retry, e.g. [502, 503, 504]

    Returns
    -------
    requests.Session
        Session retrying the POST requests to the service, over http and https

    Examples
    --------
    >>> session = resilient_session([502, 503, 504])
    >>> retry = session.get_adapter("http://service").max_retries
    >>> retry.total, retry.read, retry.status_forcelist
    (2, 0, [502, 503, 504])
    >>> session.get_adapter("https://service").max_retries is retry
    True

    The cap on Retry-After survives the copies made at each attempt

    >>> type(retry.new()).__name__
    'CappedRetry'
    """
    retry = CappedRetry(
        total=2,
        read=0,
        backoff_factor=1,
        status_forcelist=list(status_forcelist),
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


if __name__ == "__main__":
    """Execute the test suite"""
    from fink_science.tester import regular_unit_tests

    regular_unit_tests(globals())
