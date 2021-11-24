# swagger_client.AuthApi

All URIs are relative to *http://bsailn1.engr.uga.edu/api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**auth_login_create**](AuthApi.md#auth_login_create) | **POST** /auth/login |
[**auth_logout_create**](AuthApi.md#auth_logout_create) | **POST** /auth/logout | Calls Django logout method and delete the Token object assigned to the current User object.
[**auth_logout_list**](AuthApi.md#auth_logout_list) | **GET** /auth/logout | Calls Django logout method and delete the Token object assigned to the current User object.
[**auth_password_change_create**](AuthApi.md#auth_password_change_create) | **POST** /auth/password/change | Calls Django Auth SetPasswordForm save method.
[**auth_password_reset_confirm_create**](AuthApi.md#auth_password_reset_confirm_create) | **POST** /auth/password/reset/confirm | Password reset e-mail link is confirmed, therefore this resets the user&#x27;s password.
[**auth_password_reset_create**](AuthApi.md#auth_password_reset_create) | **POST** /auth/password/reset | Calls Django Auth PasswordResetForm save method.
[**auth_register_create**](AuthApi.md#auth_register_create) | **POST** /auth/register |
[**auth_signing_create**](AuthApi.md#auth_signing_create) | **POST** /auth/signing | This method signs URL for access to the server.

# **auth_login_create**
> Login auth_login_create(body)



Check the credentials and return the REST Token if the credentials are valid and authenticated. Calls Django Auth login method to register User ID in Django session framework  Accept the following POST parameters: username, password Return the REST Framework Token Object's key.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint
# Configure HTTP basic authorization: Basic
configuration = swagger_client.Configuration()
configuration.username = 'YOUR_USERNAME'
configuration.password = 'YOUR_PASSWORD'

# create an instance of the API class
api_instance = swagger_client.AuthApi(swagger_client.ApiClient(configuration))
body = swagger_client.Login() # Login |

try:
    api_response = api_instance.auth_login_create(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling AuthApi->auth_login_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Login**](Login.md)|  |

### Return type

[**Login**](Login.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **auth_logout_create**
> auth_logout_create()

Calls Django logout method and delete the Token object assigned to the current User object.

Accepts/Returns nothing.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint
# Configure HTTP basic authorization: Basic
configuration = swagger_client.Configuration()
configuration.username = 'YOUR_USERNAME'
configuration.password = 'YOUR_PASSWORD'

# create an instance of the API class
api_instance = swagger_client.AuthApi(swagger_client.ApiClient(configuration))

try:
    # Calls Django logout method and delete the Token object assigned to the current User object.
    api_instance.auth_logout_create()
except ApiException as e:
    print("Exception when calling AuthApi->auth_logout_create: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **auth_logout_list**
> auth_logout_list()

Calls Django logout method and delete the Token object assigned to the current User object.

Accepts/Returns nothing.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint
# Configure HTTP basic authorization: Basic
configuration = swagger_client.Configuration()
configuration.username = 'YOUR_USERNAME'
configuration.password = 'YOUR_PASSWORD'

# create an instance of the API class
api_instance = swagger_client.AuthApi(swagger_client.ApiClient(configuration))

try:
    # Calls Django logout method and delete the Token object assigned to the current User object.
    api_instance.auth_logout_list()
except ApiException as e:
    print("Exception when calling AuthApi->auth_logout_list: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **auth_password_change_create**
> PasswordChange auth_password_change_create(body)

Calls Django Auth SetPasswordForm save method.

Accepts the following POST parameters: new_password1, new_password2 Returns the success/fail message.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint
# Configure HTTP basic authorization: Basic
configuration = swagger_client.Configuration()
configuration.username = 'YOUR_USERNAME'
configuration.password = 'YOUR_PASSWORD'

# create an instance of the API class
api_instance = swagger_client.AuthApi(swagger_client.ApiClient(configuration))
body = swagger_client.PasswordChange() # PasswordChange |

try:
    # Calls Django Auth SetPasswordForm save method.
    api_response = api_instance.auth_password_change_create(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling AuthApi->auth_password_change_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**PasswordChange**](PasswordChange.md)|  |

### Return type

[**PasswordChange**](PasswordChange.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **auth_password_reset_confirm_create**
> PasswordResetConfirm auth_password_reset_confirm_create(body)

Password reset e-mail link is confirmed, therefore this resets the user's password.

Accepts the following POST parameters: token, uid,     new_password1, new_password2 Returns the success/fail message.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint
# Configure HTTP basic authorization: Basic
configuration = swagger_client.Configuration()
configuration.username = 'YOUR_USERNAME'
configuration.password = 'YOUR_PASSWORD'

# create an instance of the API class
api_instance = swagger_client.AuthApi(swagger_client.ApiClient(configuration))
body = swagger_client.PasswordResetConfirm() # PasswordResetConfirm |

try:
    # Password reset e-mail link is confirmed, therefore this resets the user's password.
    api_response = api_instance.auth_password_reset_confirm_create(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling AuthApi->auth_password_reset_confirm_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**PasswordResetConfirm**](PasswordResetConfirm.md)|  |

### Return type

[**PasswordResetConfirm**](PasswordResetConfirm.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **auth_password_reset_create**
> PasswordResetSerializerEx auth_password_reset_create(body)

Calls Django Auth PasswordResetForm save method.

Accepts the following POST parameters: email Returns the success/fail message.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint
# Configure HTTP basic authorization: Basic
configuration = swagger_client.Configuration()
configuration.username = 'YOUR_USERNAME'
configuration.password = 'YOUR_PASSWORD'

# create an instance of the API class
api_instance = swagger_client.AuthApi(swagger_client.ApiClient(configuration))
body = swagger_client.PasswordResetSerializerEx() # PasswordResetSerializerEx |

try:
    # Calls Django Auth PasswordResetForm save method.
    api_response = api_instance.auth_password_reset_create(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling AuthApi->auth_password_reset_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**PasswordResetSerializerEx**](PasswordResetSerializerEx.md)|  |

### Return type

[**PasswordResetSerializerEx**](PasswordResetSerializerEx.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **auth_register_create**
> RestrictedRegister auth_register_create(body)



### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint
# Configure HTTP basic authorization: Basic
configuration = swagger_client.Configuration()
configuration.username = 'YOUR_USERNAME'
configuration.password = 'YOUR_PASSWORD'

# create an instance of the API class
api_instance = swagger_client.AuthApi(swagger_client.ApiClient(configuration))
body = swagger_client.RestrictedRegister() # RestrictedRegister |

try:
    api_response = api_instance.auth_register_create(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling AuthApi->auth_register_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**RestrictedRegister**](RestrictedRegister.md)|  |

### Return type

[**RestrictedRegister**](RestrictedRegister.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **auth_signing_create**
> auth_signing_create(body)

This method signs URL for access to the server.

Signed URL contains a token which authenticates a user on the server. Signed URL is valid during 30 seconds since signing.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint
# Configure HTTP basic authorization: Basic
configuration = swagger_client.Configuration()
configuration.username = 'YOUR_USERNAME'
configuration.password = 'YOUR_PASSWORD'

# create an instance of the API class
api_instance = swagger_client.AuthApi(swagger_client.ApiClient(configuration))
body = swagger_client.AuthSigningBody() # AuthSigningBody |

try:
    # This method signs URL for access to the server.
    api_instance.auth_signing_create(body)
except ApiException as e:
    print("Exception when calling AuthApi->auth_signing_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**AuthSigningBody**](AuthSigningBody.md)|  |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)
