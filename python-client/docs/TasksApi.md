# swagger_client.TasksApi

All URIs are relative to *http://bsailn1.engr.uga.edu/api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**tasks_annotations_delete**](TasksApi.md#tasks_annotations_delete) | **DELETE** /tasks/{id}/annotations | Method deletes all annotations for a specific task
[**tasks_annotations_partial_update**](TasksApi.md#tasks_annotations_partial_update) | **PATCH** /tasks/{id}/annotations | Method performs a partial update of annotations in a specific task
[**tasks_annotations_read**](TasksApi.md#tasks_annotations_read) | **GET** /tasks/{id}/annotations | Method allows to download task annotations
[**tasks_annotations_update**](TasksApi.md#tasks_annotations_update) | **PUT** /tasks/{id}/annotations | Method allows to upload task annotations
[**tasks_create**](TasksApi.md#tasks_create) | **POST** /tasks | Method creates a new task in a database without any attached images and videos
[**tasks_data_create**](TasksApi.md#tasks_data_create) | **POST** /tasks/{id}/data | Method permanently attaches images or video to a task
[**tasks_data_data_info**](TasksApi.md#tasks_data_data_info) | **GET** /tasks/{id}/data/meta | Method provides a meta information about media files which are related with the task
[**tasks_data_read**](TasksApi.md#tasks_data_read) | **GET** /tasks/{id}/data | Method returns data for a specific task
[**tasks_dataset_export**](TasksApi.md#tasks_dataset_export) | **GET** /tasks/{id}/dataset | Export task as a dataset in a specific format
[**tasks_delete**](TasksApi.md#tasks_delete) | **DELETE** /tasks/{id} | Method deletes a specific task, all attached jobs, annotations, and data
[**tasks_jobs**](TasksApi.md#tasks_jobs) | **GET** /tasks/{id}/jobs | Returns a list of jobs for a specific task
[**tasks_list**](TasksApi.md#tasks_list) | **GET** /tasks | Returns a paginated list of tasks according to query parameters (10 tasks per page)
[**tasks_partial_update**](TasksApi.md#tasks_partial_update) | **PATCH** /tasks/{id} | Methods does a partial update of chosen fields in a task
[**tasks_read**](TasksApi.md#tasks_read) | **GET** /tasks/{id} | Method returns details of a specific task
[**tasks_status**](TasksApi.md#tasks_status) | **GET** /tasks/{id}/status | When task is being created the method returns information about a status of the creation process
[**tasks_update**](TasksApi.md#tasks_update) | **PUT** /tasks/{id} | Method updates a task by id

# **tasks_annotations_delete**
> tasks_annotations_delete(id)

Method deletes all annotations for a specific task

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this task.

try:
    # Method deletes all annotations for a specific task
    api_instance.tasks_annotations_delete(id)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_annotations_delete: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this task. |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_annotations_partial_update**
> LabeledData tasks_annotations_partial_update(body, action, id)

Method performs a partial update of annotations in a specific task

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
body = swagger_client.LabeledData() # LabeledData |
action = 'action_example' # str |
id = 56 # int | A unique integer value identifying this task.

try:
    # Method performs a partial update of annotations in a specific task
    api_response = api_instance.tasks_annotations_partial_update(body, action, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_annotations_partial_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**LabeledData**](LabeledData.md)|  |
 **action** | **str**|  |
 **id** | **int**| A unique integer value identifying this task. |

### Return type

[**LabeledData**](LabeledData.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_annotations_read**
> tasks_annotations_read(id, format=format, filename=filename, action=action)

Method allows to download task annotations

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this task.
format = 'format_example' # str | Desired output format name You can get the list of supported formats at: /server/annotation/formats (optional)
filename = 'filename_example' # str | Desired output file name (optional)
action = 'action_example' # str | Used to start downloading process after annotation file had been created (optional)

try:
    # Method allows to download task annotations
    api_instance.tasks_annotations_read(id, format=format, filename=filename, action=action)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_annotations_read: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this task. |
 **format** | **str**| Desired output format name You can get the list of supported formats at: /server/annotation/formats | [optional]
 **filename** | **str**| Desired output file name | [optional]
 **action** | **str**| Used to start downloading process after annotation file had been created | [optional]

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_annotations_update**
> tasks_annotations_update(body, id, format=format)

Method allows to upload task annotations

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
body = swagger_client.LabeledData() # LabeledData |
id = 56 # int | A unique integer value identifying this task.
format = 'format_example' # str | Input format name You can get the list of supported formats at: /server/annotation/formats (optional)

try:
    # Method allows to upload task annotations
    api_instance.tasks_annotations_update(body, id, format=format)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_annotations_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**LabeledData**](LabeledData.md)|  |
 **id** | **int**| A unique integer value identifying this task. |
 **format** | **str**| Input format name You can get the list of supported formats at: /server/annotation/formats | [optional]

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_create**
> Task tasks_create(body)

Method creates a new task in a database without any attached images and videos

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
body = swagger_client.Task() # Task |

try:
    # Method creates a new task in a database without any attached images and videos
    api_response = api_instance.tasks_create(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Task**](Task.md)|  |

### Return type

[**Task**](Task.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_data_create**
> Data tasks_data_create(body, id)

Method permanently attaches images or video to a task

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
body = swagger_client.Data() # Data |
id = 56 # int | A unique integer value identifying this task.

try:
    # Method permanently attaches images or video to a task
    api_response = api_instance.tasks_data_create(body, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_data_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Data**](Data.md)|  |
 **id** | **int**| A unique integer value identifying this task. |

### Return type

[**Data**](Data.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_data_data_info**
> DataMeta tasks_data_data_info(id)

Method provides a meta information about media files which are related with the task

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this task.

try:
    # Method provides a meta information about media files which are related with the task
    api_response = api_instance.tasks_data_data_info(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_data_data_info: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this task. |

### Return type

[**DataMeta**](DataMeta.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_data_read**
> Task tasks_data_read(id, type, quality, number)

Method returns data for a specific task

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this task.
type = 'type_example' # str | Specifies the type of the requested data
quality = 'quality_example' # str | Specifies the quality level of the requested data, doesn't matter for 'preview' type
number = 1.2 # float | A unique number value identifying chunk or frame, doesn't matter for 'preview' type

try:
    # Method returns data for a specific task
    api_response = api_instance.tasks_data_read(id, type, quality, number)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_data_read: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this task. |
 **type** | **str**| Specifies the type of the requested data |
 **quality** | **str**| Specifies the quality level of the requested data, doesn&#x27;t matter for &#x27;preview&#x27; type |
 **number** | **float**| A unique number value identifying chunk or frame, doesn&#x27;t matter for &#x27;preview&#x27; type |

### Return type

[**Task**](Task.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_dataset_export**
> tasks_dataset_export(id, format, filename=filename, action=action)

Export task as a dataset in a specific format

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this task.
format = 'format_example' # str | Desired output format name You can get the list of supported formats at: /server/annotation/formats
filename = 'filename_example' # str | Desired output file name (optional)
action = 'action_example' # str | Used to start downloading process after annotation file had been created (optional)

try:
    # Export task as a dataset in a specific format
    api_instance.tasks_dataset_export(id, format, filename=filename, action=action)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_dataset_export: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this task. |
 **format** | **str**| Desired output format name You can get the list of supported formats at: /server/annotation/formats |
 **filename** | **str**| Desired output file name | [optional]
 **action** | **str**| Used to start downloading process after annotation file had been created | [optional]

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_delete**
> tasks_delete(id)

Method deletes a specific task, all attached jobs, annotations, and data

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this task.

try:
    # Method deletes a specific task, all attached jobs, annotations, and data
    api_instance.tasks_delete(id)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_delete: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this task. |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_jobs**
> list[Job] tasks_jobs(id)

Returns a list of jobs for a specific task

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this task.

try:
    # Returns a list of jobs for a specific task
    api_response = api_instance.tasks_jobs(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_jobs: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this task. |

### Return type

[**list[Job]**](Job.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_list**
> InlineResponse2001 tasks_list(search=search, id=id, name=name, owner=owner, mode=mode, status=status, assignee=assignee, ordering=ordering, page=page, page_size=page_size)

Returns a paginated list of tasks according to query parameters (10 tasks per page)

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
search = 'search_example' # str | A search term. (optional)
id = 1.2 # float | A unique number value identifying this task (optional)
name = 'name_example' # str | Find all tasks where name contains a parameter value (optional)
owner = 'owner_example' # str | Find all tasks where owner name contains a parameter value (optional)
mode = 'mode_example' # str | Find all tasks with a specific mode (optional)
status = 'status_example' # str | Find all tasks with a specific status (optional)
assignee = 'assignee_example' # str | Find all tasks where assignee name contains a parameter value (optional)
ordering = 'ordering_example' # str | Which field to use when ordering the results. (optional)
page = 56 # int | A page number within the paginated result set. (optional)
page_size = 56 # int | Number of results to return per page. (optional)

try:
    # Returns a paginated list of tasks according to query parameters (10 tasks per page)
    api_response = api_instance.tasks_list(search=search, id=id, name=name, owner=owner, mode=mode, status=status, assignee=assignee, ordering=ordering, page=page, page_size=page_size)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_list: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **search** | **str**| A search term. | [optional]
 **id** | **float**| A unique number value identifying this task | [optional]
 **name** | **str**| Find all tasks where name contains a parameter value | [optional]
 **owner** | **str**| Find all tasks where owner name contains a parameter value | [optional]
 **mode** | **str**| Find all tasks with a specific mode | [optional]
 **status** | **str**| Find all tasks with a specific status | [optional]
 **assignee** | **str**| Find all tasks where assignee name contains a parameter value | [optional]
 **ordering** | **str**| Which field to use when ordering the results. | [optional]
 **page** | **int**| A page number within the paginated result set. | [optional]
 **page_size** | **int**| Number of results to return per page. | [optional]

### Return type

[**InlineResponse2001**](InlineResponse2001.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_partial_update**
> Task tasks_partial_update(body, id)

Methods does a partial update of chosen fields in a task

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
body = swagger_client.Task() # Task |
id = 56 # int | A unique integer value identifying this task.

try:
    # Methods does a partial update of chosen fields in a task
    api_response = api_instance.tasks_partial_update(body, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_partial_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Task**](Task.md)|  |
 **id** | **int**| A unique integer value identifying this task. |

### Return type

[**Task**](Task.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_read**
> Task tasks_read(id)

Method returns details of a specific task

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this task.

try:
    # Method returns details of a specific task
    api_response = api_instance.tasks_read(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_read: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this task. |

### Return type

[**Task**](Task.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_status**
> RqStatus tasks_status(id)

When task is being created the method returns information about a status of the creation process

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this task.

try:
    # When task is being created the method returns information about a status of the creation process
    api_response = api_instance.tasks_status(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_status: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this task. |

### Return type

[**RqStatus**](RqStatus.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_update**
> Task tasks_update(body, id)

Method updates a task by id

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
api_instance = swagger_client.TasksApi(swagger_client.ApiClient(configuration))
body = swagger_client.Task() # Task |
id = 56 # int | A unique integer value identifying this task.

try:
    # Method updates a task by id
    api_response = api_instance.tasks_update(body, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TasksApi->tasks_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Task**](Task.md)|  |
 **id** | **int**| A unique integer value identifying this task. |

### Return type

[**Task**](Task.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)
