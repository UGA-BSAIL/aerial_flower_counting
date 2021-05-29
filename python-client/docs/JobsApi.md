# swagger_client.JobsApi

All URIs are relative to *http://localhost:8080/api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**jobs_annotations_delete**](JobsApi.md#jobs_annotations_delete) | **DELETE** /jobs/{id}/annotations | Method deletes all annotations for a specific job
[**jobs_annotations_partial_update**](JobsApi.md#jobs_annotations_partial_update) | **PATCH** /jobs/{id}/annotations | Method performs a partial update of annotations in a specific job
[**jobs_annotations_read**](JobsApi.md#jobs_annotations_read) | **GET** /jobs/{id}/annotations | Method returns annotations for a specific job
[**jobs_annotations_update**](JobsApi.md#jobs_annotations_update) | **PUT** /jobs/{id}/annotations | Method performs an update of all annotations in a specific job
[**jobs_issues**](JobsApi.md#jobs_issues) | **GET** /jobs/{id}/issues | Method returns list of issues for the job
[**jobs_partial_update**](JobsApi.md#jobs_partial_update) | **PATCH** /jobs/{id} | Methods does a partial update of chosen fields in a job
[**jobs_read**](JobsApi.md#jobs_read) | **GET** /jobs/{id} | Method returns details of a job
[**jobs_reviews**](JobsApi.md#jobs_reviews) | **GET** /jobs/{id}/reviews | Method returns list of reviews for the job
[**jobs_update**](JobsApi.md#jobs_update) | **PUT** /jobs/{id} | Method updates a job by id

# **jobs_annotations_delete**
> jobs_annotations_delete(id)

Method deletes all annotations for a specific job

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
api_instance = swagger_client.JobsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this job.

try:
    # Method deletes all annotations for a specific job
    api_instance.jobs_annotations_delete(id)
except ApiException as e:
    print("Exception when calling JobsApi->jobs_annotations_delete: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this job. |

### Return type

void (empty response body)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **jobs_annotations_partial_update**
> LabeledData jobs_annotations_partial_update(body, action, id)

Method performs a partial update of annotations in a specific job

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
api_instance = swagger_client.JobsApi(swagger_client.ApiClient(configuration))
body = swagger_client.LabeledData() # LabeledData |
action = 'action_example' # str |
id = 56 # int | A unique integer value identifying this job.

try:
    # Method performs a partial update of annotations in a specific job
    api_response = api_instance.jobs_annotations_partial_update(body, action, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling JobsApi->jobs_annotations_partial_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**LabeledData**](LabeledData.md)|  |
 **action** | **str**|  |
 **id** | **int**| A unique integer value identifying this job. |

### Return type

[**LabeledData**](LabeledData.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **jobs_annotations_read**
> LabeledData jobs_annotations_read(id)

Method returns annotations for a specific job

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
api_instance = swagger_client.JobsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this job.

try:
    # Method returns annotations for a specific job
    api_response = api_instance.jobs_annotations_read(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling JobsApi->jobs_annotations_read: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this job. |

### Return type

[**LabeledData**](LabeledData.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **jobs_annotations_update**
> LabeledData jobs_annotations_update(body, id)

Method performs an update of all annotations in a specific job

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
api_instance = swagger_client.JobsApi(swagger_client.ApiClient(configuration))
body = swagger_client.LabeledData() # LabeledData |
id = 56 # int | A unique integer value identifying this job.

try:
    # Method performs an update of all annotations in a specific job
    api_response = api_instance.jobs_annotations_update(body, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling JobsApi->jobs_annotations_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**LabeledData**](LabeledData.md)|  |
 **id** | **int**| A unique integer value identifying this job. |

### Return type

[**LabeledData**](LabeledData.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **jobs_issues**
> list[CombinedIssue] jobs_issues(id)

Method returns list of issues for the job

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
api_instance = swagger_client.JobsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this job.

try:
    # Method returns list of issues for the job
    api_response = api_instance.jobs_issues(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling JobsApi->jobs_issues: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this job. |

### Return type

[**list[CombinedIssue]**](CombinedIssue.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **jobs_partial_update**
> Job jobs_partial_update(body, id)

Methods does a partial update of chosen fields in a job

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
api_instance = swagger_client.JobsApi(swagger_client.ApiClient(configuration))
body = swagger_client.Job() # Job |
id = 56 # int | A unique integer value identifying this job.

try:
    # Methods does a partial update of chosen fields in a job
    api_response = api_instance.jobs_partial_update(body, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling JobsApi->jobs_partial_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Job**](Job.md)|  |
 **id** | **int**| A unique integer value identifying this job. |

### Return type

[**Job**](Job.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **jobs_read**
> Job jobs_read(id)

Method returns details of a job

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
api_instance = swagger_client.JobsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this job.

try:
    # Method returns details of a job
    api_response = api_instance.jobs_read(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling JobsApi->jobs_read: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this job. |

### Return type

[**Job**](Job.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **jobs_reviews**
> list[Review] jobs_reviews(id)

Method returns list of reviews for the job

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
api_instance = swagger_client.JobsApi(swagger_client.ApiClient(configuration))
id = 56 # int | A unique integer value identifying this job.

try:
    # Method returns list of reviews for the job
    api_response = api_instance.jobs_reviews(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling JobsApi->jobs_reviews: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | **int**| A unique integer value identifying this job. |

### Return type

[**list[Review]**](Review.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **jobs_update**
> Job jobs_update(body, id)

Method updates a job by id

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
api_instance = swagger_client.JobsApi(swagger_client.ApiClient(configuration))
body = swagger_client.Job() # Job |
id = 56 # int | A unique integer value identifying this job.

try:
    # Method updates a job by id
    api_response = api_instance.jobs_update(body, id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling JobsApi->jobs_update: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Job**](Job.md)|  |
 **id** | **int**| A unique integer value identifying this job. |

### Return type

[**Job**](Job.md)

### Authorization

[Basic](../README.md#Basic)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)
