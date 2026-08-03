from auth.password import *

pw = "MyPassword123!"

hashed = hash_password(pw)

print(hashed)

print(verify_password(pw, hashed))

print(validate_password("abc"))

print(generate_password())

print(generate_session_token())

print(generate_csrf_token())
