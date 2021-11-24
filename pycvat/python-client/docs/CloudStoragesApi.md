# swagger_client.CloudStoragesApi

All URIs are relative to *http://bsailn1.engr.uga.edu/api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**cloudstorages_content**](CloudStoragesApi.md#cloudstorages_content) | **GET** /cloudstorages/{id}/content | Method returns a manifest content
[**cloudstorages_create**](CloudStoragesApi.md#cloudstorages_create) | **POST** /cloudstorages | Method creates a cloud storage with a specified characteristics
[**cloudstorages_delete**](CloudStoragesApi.md#cloudstorages_delete) | **DELETE** /cloudstorages/{id} | Method deletes a specific cloud storage
[**cloudstorages_list**](CloudStoragesApi.md#cloudstorages_list) | **GET** /cloudstorages | Returns a paginated list of storages according to query parameters
[**cloudstorages_partial_update**](CloudStoragesApi.md#cloudstorages_partial_update) | **PATCH** /cloudstorages/{id} | Methods does a partial update of chosen fields in a cloud storage instance
[**cloudstorages_preview**](CloudStoragesApi.md#cloudstorages_preview) | **GET** /cloudstorages/{id}/preview | Method returns a preview image from a cloud storage
[**cloudstorages_read**](CloudStoragesApi.md#cloudstorages_read) | **GET** /cloudstorages/{id} | Method returns details of a specific cloud storage
[**cloudstorages_status**](CloudStoragesApi.md#cloudstorages_status) | **GET** /cloudstorages/{id}/status | Method returns a cloud storage status

# **cloudstorages_content**
> cloudstorages_content(id, manifest_path=manifest_path)

Method returns a manifest content

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
api_instance = swagger_client.CloudStoragesApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this cloud storage.
manifest_path = 'manifest_path_example' # str | Path to the manifest file in a cloud storage (optional)

try:
    # Method returns a manifest content
    api_instance.cloudstorages_content(id, manifest_path=manifest_path)
except ApiException as e:
    print("Exception when calling CloudStoragesApi->cloudstorages_content: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this cloud storage. |
 **manifest_path** | **str**| Path to the manifest file in a cloud storage | [optional]

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **cloudstorages_create**
> cloudstorages_create(body)

Method creates a cloud storage with a specified characteristics

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
api_instance = swagger_client.CloudStoragesApi(swagger_client.ApiClient(configuration))
body = swagger_client.CloudStorage() # CloudStorage |

try:
    # Method creates a cloud storage with a specified characteristics
    api_instance.cloudstorages_create(body)
except ApiException as e:
    print("Exception when calling CloudStoragesApi->cloudstorages_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**CloudStorage**](CloudStorage.md)|  |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **cloudstorages_delete**
> cloudstorages_delete(id)

Method deletes a specific cloud storage

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
api_instance = swagger_client.CloudStoragesApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this cloud storage.

try:
    # Method deletes a specific cloud storage
    api_instance.cloudstorages_delete(id)
except ApiException as e:
    print("Exception when calling CloudStoragesApi->cloudstorages_delete: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this cloud storage. |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **cloudstorages_list**
> list[BaseCloudStorage] cloudstorages_list(search=search, id=id, display_name=display_name, provider_type=provider_type, resource=resource, credentials_type=credentials_type, description=description, owner=owner, ordering=ordering, page=page, page_size=page_size)

Returns a paginated list of storages according to query parameters

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
api_instance = swagger_client.CloudStoragesApi(swagger_client.ApiClient(configuration))
search = 'search_example' # str | A search term. (optional)
id = 1.2 # float |  (optional)
display_name = 'display_name_example' # str | A display name of storage (optional)
provider_type = 'provider_type_example' # str | A supported provider of cloud storages (optional)
resource = 'resource_example' # str | A name of bucket or container (optional)
credentials_type = 'credentials_type_example' # str | A type of a granting access (optional)
description = 'description_example' # str |  (optional)
owner = 'owner_example' # str | A resource owner (optional)
ordering = 'ordering_example' # str | Which field to use when ordering the results. (optional)
page = 56 # int | A page number within the paginated result set. (optional)
page_size = 56 # int | Number of results to return per page. (optional)

try:
    # Returns a paginated list of storages according to query parameters
    api_response = api_instance.cloudstorages_list(search=search, id=id, display_name=display_name, provider_type=provider_type, resource=resource, credentials_type=credentials_type, description=description, owner=owner, ordering=ordering, page=page, page_size=page_size)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CloudStoragesApi->cloudstorages_list: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **search** | **str**| A search term. | [optional]
 **id** | **float**|  | [optional]
 **display_name** | **str**| A display name of storage | [optional]
 **provider_type** | **str**| A supported provider of cloud storages | [optional]
 **resource** | **str**| A name of bucket or container | [optional]
 **credentials_type** | **str**| A type of a granting access | [optional]
 **description** | **str**|  | [optional]
 **owner** | **str**| A resource owner | [optional]
 **ordering** | **str**| Which field to use when ordering the results. | [optional]
 **page** | **int**| A page number within the paginated result set. | [optional]
 **page_size** | **int**| Number of results to return per page. | [optional]

### Return type

[**list[BaseCloudStorage]**](BaseCloudStorage.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **cloudstorages_partial_update**
> CloudStorage cloudstorages_partial_update(body, id)

Methods does a partial update of chosen fields in a cloud storage instance

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
api_instance = swagger_client.CloudStoragesApi(swagger_client.ApiClient(configuration))
body = swagger_client.CloudStorage() # CloudStorage |
id = 56 # int | A unique integer value identifying this cloud storage.

try:
    # Methods does a partial update of chosen fields in a cloud storage instance
    api_response = api_instance.cloudstorages_partial_update(body, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CloudStoragesApi->cloudstorages_partial_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**CloudStorage**](CloudStorage.md)|  |
 **id** | **int**| A unique integer value identifying this cloud storage. |

### Return type

[**CloudStorage**](CloudStorage.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **cloudstorages_preview**
> cloudstorages_preview(id)

Method returns a preview image from a cloud storage

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
api_instance = swagger_client.CloudStoragesApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this cloud storage.

try:
    # Method returns a preview image from a cloud storage
    api_instance.cloudstorages_preview(id)
except ApiException as e:
    print("Exception when calling CloudStoragesApi->cloudstorages_preview: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this cloud storage. |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **cloudstorages_read**
> cloudstorages_read(id)

Method returns details of a specific cloud storage

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
api_instance = swagger_client.CloudStoragesApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this cloud storage.

try:
    # Method returns details of a specific cloud storage
    api_instance.cloudstorages_read(id)
except ApiException as e:
    print("Exception when calling CloudStoragesApi->cloudstorages_read: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this cloud storage. |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **cloudstorages_status**
> cloudstorages_status(id)

Method returns a cloud storage status

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
api_instance = swagger_client.CloudStoragesApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this cloud storage.

try:
    # Method returns a cloud storage status
    api_instance.cloudstorages_status(id)
except ApiException as e:
    print("Exception when calling CloudStoragesApi->cloudstorages_status: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this cloud storage. |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)
