# coding: utf-8

"""
    CVAT REST API

    REST API for Computer Vision Annotation Tool (CVAT)  # noqa: E501

    OpenAPI spec version: v1
    Contact: nikita.manovich@intel.com
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

from __future__ import absolute_import

import re  # noqa: F401

# python 2 and python 3 compatibility library
import six
from swagger_client.api_client import ApiClient


class JobsApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def jobs_annotations_delete(self, id, **kwargs):  # noqa: E501
        """Method deletes all annotations for a specific job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_annotations_delete(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this job. (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.jobs_annotations_delete_with_http_info(
                id, **kwargs
            )  # noqa: E501
        else:
            (data) = self.jobs_annotations_delete_with_http_info(
                id, **kwargs
            )  # noqa: E501
            return data

    def jobs_annotations_delete_with_http_info(
        self, id, **kwargs
    ):  # noqa: E501
        """Method deletes all annotations for a specific job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_annotations_delete_with_http_info(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this job. (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ["id"]  # noqa: E501
        all_params.append("async_req")
        all_params.append("_return_http_data_only")
        all_params.append("_preload_content")
        all_params.append("_request_timeout")

        params = locals()
        for key, val in six.iteritems(params["kwargs"]):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method jobs_annotations_delete" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `jobs_annotations_delete`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in params:
            path_params["id"] = params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # Authentication setting
        auth_settings = ["Basic"]  # noqa: E501

        return self.api_client.call_api(
            "/jobs/{id}/annotations",
            "DELETE",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type=None,  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def jobs_annotations_partial_update(
        self, body, action, id, **kwargs
    ):  # noqa: E501
        """Method performs a partial update of annotations in a specific job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_annotations_partial_update(body, action, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param LabeledData body: (required)
        :param str action: (required)
        :param int id: A unique integer value identifying this job. (required)
        :return: LabeledData
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.jobs_annotations_partial_update_with_http_info(
                body, action, id, **kwargs
            )  # noqa: E501
        else:
            (data) = self.jobs_annotations_partial_update_with_http_info(
                body, action, id, **kwargs
            )  # noqa: E501
            return data

    def jobs_annotations_partial_update_with_http_info(
        self, body, action, id, **kwargs
    ):  # noqa: E501
        """Method performs a partial update of annotations in a specific job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_annotations_partial_update_with_http_info(body, action, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param LabeledData body: (required)
        :param str action: (required)
        :param int id: A unique integer value identifying this job. (required)
        :return: LabeledData
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ["body", "action", "id"]  # noqa: E501
        all_params.append("async_req")
        all_params.append("_return_http_data_only")
        all_params.append("_preload_content")
        all_params.append("_request_timeout")

        params = locals()
        for key, val in six.iteritems(params["kwargs"]):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method jobs_annotations_partial_update" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'body' is set
        if "body" not in params or params["body"] is None:
            raise ValueError(
                "Missing the required parameter `body` when calling `jobs_annotations_partial_update`"
            )  # noqa: E501
        # verify the required parameter 'action' is set
        if "action" not in params or params["action"] is None:
            raise ValueError(
                "Missing the required parameter `action` when calling `jobs_annotations_partial_update`"
            )  # noqa: E501
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `jobs_annotations_partial_update`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in params:
            path_params["id"] = params["id"]  # noqa: E501

        query_params = []
        if "action" in params:
            query_params.append(("action", params["action"]))  # noqa: E501

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if "body" in params:
            body_params = params["body"]
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json"]
        )  # noqa: E501

        # HTTP header `Content-Type`
        header_params[
            "Content-Type"
        ] = self.api_client.select_header_content_type(  # noqa: E501
            ["application/json"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = ["Basic"]  # noqa: E501

        return self.api_client.call_api(
            "/jobs/{id}/annotations",
            "PATCH",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="LabeledData",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def jobs_annotations_read(self, id, **kwargs):  # noqa: E501
        """Method returns annotations for a specific job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_annotations_read(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this job. (required)
        :return: LabeledData
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.jobs_annotations_read_with_http_info(
                id, **kwargs
            )  # noqa: E501
        else:
            (data) = self.jobs_annotations_read_with_http_info(
                id, **kwargs
            )  # noqa: E501
            return data

    def jobs_annotations_read_with_http_info(self, id, **kwargs):  # noqa: E501
        """Method returns annotations for a specific job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_annotations_read_with_http_info(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this job. (required)
        :return: LabeledData
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ["id"]  # noqa: E501
        all_params.append("async_req")
        all_params.append("_return_http_data_only")
        all_params.append("_preload_content")
        all_params.append("_request_timeout")

        params = locals()
        for key, val in six.iteritems(params["kwargs"]):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method jobs_annotations_read" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `jobs_annotations_read`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in params:
            path_params["id"] = params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = ["Basic"]  # noqa: E501

        return self.api_client.call_api(
            "/jobs/{id}/annotations",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="LabeledData",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def jobs_annotations_update(self, body, id, **kwargs):  # noqa: E501
        """Method performs an update of all annotations in a specific job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_annotations_update(body, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param LabeledData body: (required)
        :param int id: A unique integer value identifying this job. (required)
        :return: LabeledData
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.jobs_annotations_update_with_http_info(
                body, id, **kwargs
            )  # noqa: E501
        else:
            (data) = self.jobs_annotations_update_with_http_info(
                body, id, **kwargs
            )  # noqa: E501
            return data

    def jobs_annotations_update_with_http_info(
        self, body, id, **kwargs
    ):  # noqa: E501
        """Method performs an update of all annotations in a specific job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_annotations_update_with_http_info(body, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param LabeledData body: (required)
        :param int id: A unique integer value identifying this job. (required)
        :return: LabeledData
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ["body", "id"]  # noqa: E501
        all_params.append("async_req")
        all_params.append("_return_http_data_only")
        all_params.append("_preload_content")
        all_params.append("_request_timeout")

        params = locals()
        for key, val in six.iteritems(params["kwargs"]):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method jobs_annotations_update" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'body' is set
        if "body" not in params or params["body"] is None:
            raise ValueError(
                "Missing the required parameter `body` when calling `jobs_annotations_update`"
            )  # noqa: E501
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `jobs_annotations_update`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in params:
            path_params["id"] = params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if "body" in params:
            body_params = params["body"]
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json"]
        )  # noqa: E501

        # HTTP header `Content-Type`
        header_params[
            "Content-Type"
        ] = self.api_client.select_header_content_type(  # noqa: E501
            ["application/json"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = ["Basic"]  # noqa: E501

        return self.api_client.call_api(
            "/jobs/{id}/annotations",
            "PUT",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="LabeledData",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def jobs_issues(self, id, **kwargs):  # noqa: E501
        """Method returns list of issues for the job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_issues(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this job. (required)
        :return: list[CombinedIssue]
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.jobs_issues_with_http_info(id, **kwargs)  # noqa: E501
        else:
            (data) = self.jobs_issues_with_http_info(
                id, **kwargs
            )  # noqa: E501
            return data

    def jobs_issues_with_http_info(self, id, **kwargs):  # noqa: E501
        """Method returns list of issues for the job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_issues_with_http_info(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this job. (required)
        :return: list[CombinedIssue]
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ["id"]  # noqa: E501
        all_params.append("async_req")
        all_params.append("_return_http_data_only")
        all_params.append("_preload_content")
        all_params.append("_request_timeout")

        params = locals()
        for key, val in six.iteritems(params["kwargs"]):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method jobs_issues" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `jobs_issues`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in params:
            path_params["id"] = params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = ["Basic"]  # noqa: E501

        return self.api_client.call_api(
            "/jobs/{id}/issues",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="list[CombinedIssue]",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def jobs_partial_update(self, body, id, **kwargs):  # noqa: E501
        """Methods does a partial update of chosen fields in a job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_partial_update(body, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param Job body: (required)
        :param int id: A unique integer value identifying this job. (required)
        :return: Job
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.jobs_partial_update_with_http_info(
                body, id, **kwargs
            )  # noqa: E501
        else:
            (data) = self.jobs_partial_update_with_http_info(
                body, id, **kwargs
            )  # noqa: E501
            return data

    def jobs_partial_update_with_http_info(
        self, body, id, **kwargs
    ):  # noqa: E501
        """Methods does a partial update of chosen fields in a job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_partial_update_with_http_info(body, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param Job body: (required)
        :param int id: A unique integer value identifying this job. (required)
        :return: Job
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ["body", "id"]  # noqa: E501
        all_params.append("async_req")
        all_params.append("_return_http_data_only")
        all_params.append("_preload_content")
        all_params.append("_request_timeout")

        params = locals()
        for key, val in six.iteritems(params["kwargs"]):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method jobs_partial_update" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'body' is set
        if "body" not in params or params["body"] is None:
            raise ValueError(
                "Missing the required parameter `body` when calling `jobs_partial_update`"
            )  # noqa: E501
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `jobs_partial_update`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in params:
            path_params["id"] = params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if "body" in params:
            body_params = params["body"]
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json"]
        )  # noqa: E501

        # HTTP header `Content-Type`
        header_params[
            "Content-Type"
        ] = self.api_client.select_header_content_type(  # noqa: E501
            ["application/json"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = ["Basic"]  # noqa: E501

        return self.api_client.call_api(
            "/jobs/{id}",
            "PATCH",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="Job",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def jobs_read(self, id, **kwargs):  # noqa: E501
        """Method returns details of a job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_read(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this job. (required)
        :return: Job
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.jobs_read_with_http_info(id, **kwargs)  # noqa: E501
        else:
            (data) = self.jobs_read_with_http_info(id, **kwargs)  # noqa: E501
            return data

    def jobs_read_with_http_info(self, id, **kwargs):  # noqa: E501
        """Method returns details of a job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_read_with_http_info(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this job. (required)
        :return: Job
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ["id"]  # noqa: E501
        all_params.append("async_req")
        all_params.append("_return_http_data_only")
        all_params.append("_preload_content")
        all_params.append("_request_timeout")

        params = locals()
        for key, val in six.iteritems(params["kwargs"]):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method jobs_read" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `jobs_read`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in params:
            path_params["id"] = params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = ["Basic"]  # noqa: E501

        return self.api_client.call_api(
            "/jobs/{id}",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="Job",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def jobs_reviews(self, id, **kwargs):  # noqa: E501
        """Method returns list of reviews for the job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_reviews(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this job. (required)
        :return: list[Review]
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.jobs_reviews_with_http_info(id, **kwargs)  # noqa: E501
        else:
            (data) = self.jobs_reviews_with_http_info(
                id, **kwargs
            )  # noqa: E501
            return data

    def jobs_reviews_with_http_info(self, id, **kwargs):  # noqa: E501
        """Method returns list of reviews for the job  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_reviews_with_http_info(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this job. (required)
        :return: list[Review]
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ["id"]  # noqa: E501
        all_params.append("async_req")
        all_params.append("_return_http_data_only")
        all_params.append("_preload_content")
        all_params.append("_request_timeout")

        params = locals()
        for key, val in six.iteritems(params["kwargs"]):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method jobs_reviews" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `jobs_reviews`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in params:
            path_params["id"] = params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = ["Basic"]  # noqa: E501

        return self.api_client.call_api(
            "/jobs/{id}/reviews",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="list[Review]",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def jobs_update(self, body, id, **kwargs):  # noqa: E501
        """Method updates a job by id  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_update(body, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param Job body: (required)
        :param int id: A unique integer value identifying this job. (required)
        :return: Job
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.jobs_update_with_http_info(
                body, id, **kwargs
            )  # noqa: E501
        else:
            (data) = self.jobs_update_with_http_info(
                body, id, **kwargs
            )  # noqa: E501
            return data

    def jobs_update_with_http_info(self, body, id, **kwargs):  # noqa: E501
        """Method updates a job by id  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.jobs_update_with_http_info(body, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param Job body: (required)
        :param int id: A unique integer value identifying this job. (required)
        :return: Job
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ["body", "id"]  # noqa: E501
        all_params.append("async_req")
        all_params.append("_return_http_data_only")
        all_params.append("_preload_content")
        all_params.append("_request_timeout")

        params = locals()
        for key, val in six.iteritems(params["kwargs"]):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method jobs_update" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'body' is set
        if "body" not in params or params["body"] is None:
            raise ValueError(
                "Missing the required parameter `body` when calling `jobs_update`"
            )  # noqa: E501
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `jobs_update`"
            )  # noqa: E501

        collection_formats = {}

        path_params = {}
        if "id" in params:
            path_params["id"] = params["id"]  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if "body" in params:
            body_params = params["body"]
        # HTTP header `Accept`
        header_params["Accept"] = self.api_client.select_header_accept(
            ["application/json"]
        )  # noqa: E501

        # HTTP header `Content-Type`
        header_params[
            "Content-Type"
        ] = self.api_client.select_header_content_type(  # noqa: E501
            ["application/json"]
        )  # noqa: E501

        # Authentication setting
        auth_settings = ["Basic"]  # noqa: E501

        return self.api_client.call_api(
            "/jobs/{id}",
            "PUT",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="Job",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )