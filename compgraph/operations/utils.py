from datetime import datetime


def parse_datetime(date: str) -> datetime:
    try:
        return datetime.strptime(date, '%Y%m%dT%H%M%S.%f')
    except ValueError:
        return datetime.strptime(date, '%Y%m%dT%H%M%S')
