# swagger_client.ReviewsApi

All URIs are relative to *http://bsailn1.engr.uga.edu/api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**reviews_create**](ReviewsApi.md#reviews_create) | **POST** /reviews | Submit a review for a job
[**reviews_delete**](ReviewsApi.md#reviews_delete) | **DELETE** /reviews/{id} | Method removes a review from a job

# **reviews_create**
> CombinedReview reviews_create(body)

Submit a review for a job

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
api_instance = swagger_client.ReviewsApi(swagger_client.ApiClient(configuration))
body = swagger_client.CombinedReview() # CombinedReview |

try:
    # Submit a review for a job
    api_response = api_instance.reviews_create(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ReviewsApi->reviews_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**CombinedReview**](CombinedReview.md)|  |

### Return type

[**CombinedReview**](CombinedReview.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **reviews_delete**
> reviews_delete(id)

Method removes a review from a job

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
api_instance = swagger_client.ReviewsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this review.

try:
    # Method removes a review from a job
    api_instance.reviews_delete(id)
except ApiException as e:
    print("Exception when calling ReviewsApi->reviews_delete: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this review. |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)
