# swagger_client.ProjectsApi

All URIs are relative to *http://bsailn1.engr.uga.edu/api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**projects_annotations**](ProjectsApi.md#projects_annotations) | **GET** /projects/{id}/annotations | Method allows to download project annotations
[**projects_create**](ProjectsApi.md#projects_create) | **POST** /projects | Method creates a new project
[**projects_dataset_export**](ProjectsApi.md#projects_dataset_export) | **GET** /projects/{id}/dataset | Export project as a dataset in a specific format
[**projects_delete**](ProjectsApi.md#projects_delete) | **DELETE** /projects/{id} | Method deletes a specific project
[**projects_list**](ProjectsApi.md#projects_list) | **GET** /projects | Returns a paginated list of projects according to query parameters (12 projects per page)
[**projects_partial_update**](ProjectsApi.md#projects_partial_update) | **PATCH** /projects/{id} | Methods does a partial update of chosen fields in a project
[**projects_read**](ProjectsApi.md#projects_read) | **GET** /projects/{id} | Method returns details of a specific project
[**projects_tasks**](ProjectsApi.md#projects_tasks) | **GET** /projects/{id}/tasks | Returns information of the tasks of the project with the selected id

# **projects_annotations**
> projects_annotations(id, format, filename=filename, action=action)

Method allows to download project annotations

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
api_instance = swagger_client.ProjectsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this project.
format = 'format_example' # str | Desired output format name You can get the list of supported formats at: /server/annotation/formats
filename = 'filename_example' # str | Desired output file name (optional)
action = 'action_example' # str | Used to start downloading process after annotation file had been created (optional)

try:
    # Method allows to download project annotations
    api_instance.projects_annotations(id, format, filename=filename, action=action)
except ApiException as e:
    print("Exception when calling ProjectsApi->projects_annotations: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this project. |
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

# **projects_create**
> Project projects_create(body)

Method creates a new project

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
api_instance = swagger_client.ProjectsApi(swagger_client.ApiClient(configuration))
body = swagger_client.Project() # Project |

try:
    # Method creates a new project
    api_response = api_instance.projects_create(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ProjectsApi->projects_create: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Project**](Project.md)|  |

### Return type

[**Project**](Project.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **projects_dataset_export**
> projects_dataset_export(id, format, filename=filename, action=action)

Export project as a dataset in a specific format

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
api_instance = swagger_client.ProjectsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this project.
format = 'format_example' # str | Desired output format name You can get the list of supported formats at: /server/annotation/formats
filename = 'filename_example' # str | Desired output file name (optional)
action = 'action_example' # str | Used to start downloading process after annotation file had been created (optional)

try:
    # Export project as a dataset in a specific format
    api_instance.projects_dataset_export(id, format, filename=filename, action=action)
except ApiException as e:
    print("Exception when calling ProjectsApi->projects_dataset_export: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this project. |
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

# **projects_delete**
> projects_delete(id)

Method deletes a specific project

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
api_instance = swagger_client.ProjectsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this project.

try:
    # Method deletes a specific project
    api_instance.projects_delete(id)
except ApiException as e:
    print("Exception when calling ProjectsApi->projects_delete: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this project. |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **projects_list**
> InlineResponse200 projects_list(search=search, id=id, name=name, owner=owner, status=status, assignee=assignee, ordering=ordering, page=page, page_size=page_size, names_only=names_only, without_tasks=without_tasks)

Returns a paginated list of projects according to query parameters (12 projects per page)

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
api_instance = swagger_client.ProjectsApi(swagger_client.ApiClient(configuration))
search = 'search_example' # str | A search term. (optional)
id = 1.2 # float | A unique number value identifying this project (optional)
name = 'name_example' # str | Find all projects where name contains a parameter value (optional)
owner = 'owner_example' # str | Find all project where owner name contains a parameter value (optional)
status = 'status_example' # str | Find all projects with a specific status (optional)
assignee = 'assignee_example' # str |  (optional)
ordering = 'ordering_example' # str | Which field to use when ordering the results. (optional)
page = 56 # int | A page number within the paginated result set. (optional)
page_size = 56 # int | Number of results to return per page. (optional)
names_only = true # bool | Returns only names and id's of projects. (optional)
without_tasks = true # bool | Returns only projects entities without related tasks (optional)

try:
    # Returns a paginated list of projects according to query parameters (12 projects per page)
    api_response = api_instance.projects_list(search=search, id=id, name=name, owner=owner, status=status, assignee=assignee, ordering=ordering, page=page, page_size=page_size, names_only=names_only, without_tasks=without_tasks)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ProjectsApi->projects_list: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **search** | **str**| A search term. | [optional]
 **id** | **float**| A unique number value identifying this project | [optional]
 **name** | **str**| Find all projects where name contains a parameter value | [optional]
 **owner** | **str**| Find all project where owner name contains a parameter value | [optional]
 **status** | **str**| Find all projects with a specific status | [optional]
 **assignee** | **str**|  | [optional]
 **ordering** | **str**| Which field to use when ordering the results. | [optional]
 **page** | **int**| A page number within the paginated result set. | [optional]
 **page_size** | **int**| Number of results to return per page. | [optional]
 **names_only** | **bool**| Returns only names and id&#x27;s of projects. | [optional]
 **without_tasks** | **bool**| Returns only projects entities without related tasks | [optional]

### Return type

[**InlineResponse200**](InlineResponse200.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **projects_partial_update**
> Project projects_partial_update(body, id)

Methods does a partial update of chosen fields in a project

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
api_instance = swagger_client.ProjectsApi(swagger_client.ApiClient(configuration))
body = swagger_client.Project() # Project |
id = 56 # int | A unique integer value identifying this project.

try:
    # Methods does a partial update of chosen fields in a project
    api_response = api_instance.projects_partial_update(body, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ProjectsApi->projects_partial_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Project**](Project.md)|  |
 **id** | **int**| A unique integer value identifying this project. |

### Return type

[**Project**](Project.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **projects_read**
> Project projects_read(id)

Method returns details of a specific project

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
api_instance = swagger_client.ProjectsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this project.

try:
    # Method returns details of a specific project
    api_response = api_instance.projects_read(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ProjectsApi->projects_read: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this project. |

### Return type

[**Project**](Project.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **projects_tasks**
> list[Task] projects_tasks(id)

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
api_instance = swagger_client.ProjectsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this project.

try:
    # Returns information of the tasks of the project with the selected id
    api_response = api_instance.projects_tasks(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ProjectsApi->projects_tasks: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this project. |

### Return type

[**list[Task]**](Task.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)
