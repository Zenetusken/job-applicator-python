from http.cookiejar import CookieJar
from typing import Callable

chrome: Callable[..., CookieJar]
chromium: Callable[..., CookieJar]
brave: Callable[..., CookieJar]
edge: Callable[..., CookieJar]
firefox: Callable[..., CookieJar]
