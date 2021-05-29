# swagger_client.ServerApi

All URIs are relative to *http://localhost:8080/api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**server_about**](ServerApi.md#server_about) | **GET** /server/about | Method provides basic CVAT information
[**server_annotation_annotation_formats**](ServerApi.md#server_annotation_annotation_formats) | **GET** /server/annotation/formats | Method provides the list of supported annotations formats
[**server_exception**](ServerApi.md#server_exception) | **POST** /server/exception | Saves an exception from a client on the server
[**server_logs**](ServerApi.md#server_logs) | **POST** /server/logs | Saves logs from a client on the server
[**server_plugins**](ServerApi.md#server_plugins) | **GET** /server/plugins | Method provides allowed plugins.
[**server_share**](ServerApi.md#server_share) | **GET** /server/share | Returns all files and folders that are on the server along specified path

# **server_about**
> About server_about()

Method provides basic CVAT information

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
api_instance = swagger_client.ServerApi(swagger_client.ApiClient(configuration))

try:
    # Method provides basic CVAT information
    api_response = api_instance.server_about()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ServerApi->server_about: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**About**](About.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **server_annotation_annotation_formats**
> DatasetFormats server_annotation_annotation_formats()

Method provides the list of supported annotations formats

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
api_instance = swagger_client.ServerApi(swagger_client.ApiClient(configuration))

try:
    # Method provides the list of supported annotations formats
    api_response = api_instance.server_annotation_annotation_formats()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ServerApi->server_annotation_annotation_formats: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**DatasetFormats**](DatasetFormats.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **server_exception**
> Exception server_exception(body)

Saves an exception from a client on the server

Sends logs to the ELK if it is connected

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
api_instance = swagger_client.ServerApi(swagger_client.ApiClient(configuration))
body = swagger_client.Exception() # Exception |

try:
    # Saves an exception from a client on the server
    api_response = api_instance.server_exception(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ServerApi->server_exception: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Exception**](Exception.md)|  |

### Return type

[**Exception**](Exception.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **server_logs**
> list[LogEvent] server_logs(body)

Saves logs from a client on the server

Sends logs to the ELK if it is connected

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
api_instance = swagger_client.ServerApi(swagger_client.ApiClient(configuration))
body = [swagger_client.LogEvent()] # list[LogEvent] |

try:
    # Saves logs from a client on the server
    api_response = api_instance.server_logs(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ServerApi->server_logs: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**list[LogEvent]**](LogEvent.md)|  |

### Return type

[**list[LogEvent]**](LogEvent.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **server_plugins**
> Plugins server_plugins()

Method provides allowed plugins.

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
api_instance = swagger_client.ServerApi(swagger_client.ApiClient(configuration))

try:
    # Method provides allowed plugins.
    api_response = api_instance.server_plugins()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ServerApi->server_plugins: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**Plugins**](Plugins.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **server_share**
> list[FileInfo] server_share(directory=directory)

Returns all files and folders that are on the server along specified path

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
api_instance = swagger_client.ServerApi(swagger_client.ApiClient(configuration))
directory = 'directory_example' # str | Directory to browse (optional)

try:
    # Returns all files and folders that are on the server along specified path
    api_response = api_instance.server_share(directory=directory)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ServerApi->server_share: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **directory** | **str**| Directory to browse | [optional]

### Return type

[**list[FileInfo]**](FileInfo.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)
