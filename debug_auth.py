from service.career_market.utils.auth_utils import _find_user_by_token
from service.career_market.utils.profile_utils import _load_profile_for_email

token = 'uJgg9C3NEfqJDmgREkG0CPHih6P5mRJlmfIXmK7T-TU'
print('Checking token:', token[:8] + '...')
user = _find_user_by_token(token)
print('User lookup result:', user)

print('\nLoading profile for email:')
profile = _load_profile_for_email('test+dev@example.com')
print(profile)
