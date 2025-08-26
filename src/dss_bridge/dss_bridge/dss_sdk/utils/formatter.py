def format_version(version: int) -> str:
    version_str = str(version).zfill(3)  # 3자리 보장
    digits = list(map(int, version_str))
    if len(digits) == 3:
        return f"{digits[0]}.{digits[1]}.{digits[2]}"
    elif len(digits) == 4:
        return f"{digits[0]}{digits[1]}.{digits[2]}.{digits[3]}"
    elif len(digits) == 5:
        return f"{digits[0]}{digits[1]}.{digits[2]}{digits[3]}.{digits[4]}"
    else:
        raise ValueError("Unsupported version format")