# swagger_client.RestrictionsApi

All URIs are relative to *http://localhost:8080/api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**restrictions_terms_of_use**](RestrictionsApi.md#restrictions_terms_of_use) | **GET** /restrictions/terms-of-use |
[**restrictions_user_agreements**](RestrictionsApi.md#restrictions_user_agreements) | **GET** /restrictions/user-agreements | Method provides user agreements that the user must accept to register

# **restrictions_terms_of_use**
> restrictions_terms_of_use()



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
api_instance = swagger_client.RestrictionsApi(swagger_client.ApiClient(configuration))

try:
    api_instance.restrictions_terms_of_use()
except ApiException as e:
    print("Exception when calling RestrictionsApi->restrictions_terms_of_use: %s\n" % e)
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

# **restrictions_user_agreements**
> UserAgreement restrictions_user_agreements()

Method provides user agreements that the user must accept to register

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
api_instance = swagger_client.RestrictionsApi(swagger_client.ApiClient(configuration))

try:
    # Method provides user agreements that the user must accept to register
    api_response = api_instance.restrictions_user_agreements()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling RestrictionsApi->restrictions_user_agreements: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**UserAgreement**](UserAgreement.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)
