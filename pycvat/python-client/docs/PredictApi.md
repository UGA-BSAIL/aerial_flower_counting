# swagger_client.PredictApi

All URIs are relative to *http://bsailn1.engr.uga.edu/api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**predict_predict_image**](PredictApi.md#predict_predict_image) | **GET** /predict/frame | Returns prediction for image
[**predict_predict_status**](PredictApi.md#predict_predict_status) | **GET** /predict/status | Returns information of the tasks of the project with the selected id

# **predict_predict_image**
> predict_predict_image()

Returns prediction for image

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
api_instance = swagger_client.PredictApi(swagger_client.ApiClient(configuration))

try:
    # Returns prediction for image
    api_instance.predict_predict_image()
except ApiException as e:
    print("Exception when calling PredictApi->predict_predict_image: %s\n" % e)
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

# **predict_predict_status**
> predict_predict_status()

Returns information of the tasks of the project with the selected id

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
api_instance = swagger_client.PredictApi(swagger_client.ApiClient(configuration))

try:
    # Returns information of the tasks of the project with the selected id
    api_instance.predict_predict_status()
except ApiException as e:
    print("Exception when calling PredictApi->predict_predict_status: %s\n" % e)
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
