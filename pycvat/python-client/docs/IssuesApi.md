# swagger_client.IssuesApi

All URIs are relative to *http://bsailn1.engr.uga.edu/api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**issues_comments**](IssuesApi.md#issues_comments) | **GET** /issues/{id}/comments | The action returns all comments of a specific issue
[**issues_delete**](IssuesApi.md#issues_delete) | **DELETE** /issues/{id} | Method removes an issue from a job
[**issues_partial_update**](IssuesApi.md#issues_partial_update) | **PATCH** /issues/{id} | Method updates an issue. It is used to resolve/reopen an issue

# **issues_comments**
> list[Comment] issues_comments(id)

The action returns all comments of a specific issue

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
api_instance = swagger_client.IssuesApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this issue.

try:
    # The action returns all comments of a specific issue
    api_response = api_instance.issues_comments(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling IssuesApi->issues_comments: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this issue. |

### Return type

[**list[Comment]**](Comment.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **issues_delete**
> issues_delete(id)

Method removes an issue from a job

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
api_instance = swagger_client.IssuesApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this issue.

try:
    # Method removes an issue from a job
    api_instance.issues_delete(id)
except ApiException as e:
    print("Exception when calling IssuesApi->issues_delete: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this issue. |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **issues_partial_update**
> Issue issues_partial_update(body, id)

Method updates an issue. It is used to resolve/reopen an issue

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
api_instance = swagger_client.IssuesApi(swagger_client.ApiClient(configuration))
body = swagger_client.Issue() # Issue |
id = 56 # int | A unique integer value identifying this issue.

try:
    # Method updates an issue. It is used to resolve/reopen an issue
    api_response = api_instance.issues_partial_update(body, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling IssuesApi->issues_partial_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Issue**](Issue.md)|  |
 **id** | **int**| A unique integer value identifying this issue. |

### Return type

[**Issue**](Issue.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)
