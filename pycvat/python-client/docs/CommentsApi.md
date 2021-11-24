# swagger_client.CommentsApi

All URIs are relative to *http://bsailn1.engr.uga.edu/api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**comments_create**](CommentsApi.md#comments_create) | **POST** /comments |
[**comments_delete**](CommentsApi.md#comments_delete) | **DELETE** /comments/{id} | Method removes a comment from an issue
[**comments_partial_update**](CommentsApi.md#comments_partial_update) | **PATCH** /comments/{id} | Method updates comment in an issue

# **comments_create**
> Comment comments_create(body)



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
api_instance = swagger_client.CommentsApi(swagger_client.ApiClient(configuration))
body = swagger_client.Comment() # Comment |

try:
    api_response = api_instance.comments_create(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CommentsApi->comments_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Comment**](Comment.md)|  |

### Return type

[**Comment**](Comment.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **comments_delete**
> comments_delete(id)

Method removes a comment from an issue

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
api_instance = swagger_client.CommentsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this comment.

try:
    # Method removes a comment from an issue
    api_instance.comments_delete(id)
except ApiException as e:
    print("Exception when calling CommentsApi->comments_delete: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this comment. |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **comments_partial_update**
> Comment comments_partial_update(body, id)

Method updates comment in an issue

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
api_instance = swagger_client.CommentsApi(swagger_client.ApiClient(configuration))
body = swagger_client.Comment() # Comment |
id = 56 # int | A unique integer value identifying this comment.

try:
    # Method updates comment in an issue
    api_response = api_instance.comments_partial_update(body, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CommentsApi->comments_partial_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Comment**](Comment.md)|  |
 **id** | **int**| A unique integer value identifying this comment. |

### Return type

[**Comment**](Comment.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)
