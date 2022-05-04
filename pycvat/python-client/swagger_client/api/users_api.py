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


class UsersApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def users_delete(self, id, **kwargs):  # noqa: E501
        """Method deletes a specific user from the server  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.users_delete(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this user. (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.users_delete_with_http_info(id, **kwargs)  # noqa: E501
        else:
            (data) = self.users_delete_with_http_info(
                id, **kwargs
            )  # noqa: E501
            return data

    def users_delete_with_http_info(self, id, **kwargs):  # noqa: E501
        """Method deletes a specific user from the server  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.users_delete_with_http_info(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this user. (required)
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
                    " to method users_delete" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `users_delete`"
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
            "/users/{id}",
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

    def users_list(self, **kwargs):  # noqa: E501
        """Method provides a paginated list of users registered on the server  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.users_list(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str search: A search term.
        :param float id: A unique number value identifying this user
        :param bool is_active: Returns only active users
        :param str ordering: Which field to use when ordering the results.
        :param int page: A page number within the paginated result set.
        :param int page_size: Number of results to return per page.
        :return: InlineResponse2002
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.users_list_with_http_info(**kwargs)  # noqa: E501
        else:
            (data) = self.users_list_with_http_info(**kwargs)  # noqa: E501
            return data

    def users_list_with_http_info(self, **kwargs):  # noqa: E501
        """Method provides a paginated list of users registered on the server  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.users_list_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str search: A search term.
        :param float id: A unique number value identifying this user
        :param bool is_active: Returns only active users
        :param str ordering: Which field to use when ordering the results.
        :param int page: A page number within the paginated result set.
        :param int page_size: Number of results to return per page.
        :return: InlineResponse2002
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = [
            "search",
            "id",
            "is_active",
            "ordering",
            "page",
            "page_size",
        ]  # noqa: E501
        all_params.append("async_req")
        all_params.append("_return_http_data_only")
        all_params.append("_preload_content")
        all_params.append("_request_timeout")

        params = locals()
        for key, val in six.iteritems(params["kwargs"]):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method users_list" % key
                )
            params[key] = val
        del params["kwargs"]

        collection_formats = {}

        path_params = {}

        query_params = []
        if "search" in params:
            query_params.append(("search", params["search"]))  # noqa: E501
        if "id" in params:
            query_params.append(("id", params["id"]))  # noqa: E501
        if "is_active" in params:
            query_params.append(
                ("is_active", params["is_active"])
            )  # noqa: E501
        if "ordering" in params:
            query_params.append(("ordering", params["ordering"]))  # noqa: E501
        if "page" in params:
            query_params.append(("page", params["page"]))  # noqa: E501
        if "page_size" in params:
            query_params.append(
                ("page_size", params["page_size"])
            )  # noqa: E501

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
            "/users",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="InlineResponse2002",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def users_partial_update(self, body, id, **kwargs):  # noqa: E501
        """Method updates chosen fields of a user  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.users_partial_update(body, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param User body: (required)
        :param int id: A unique integer value identifying this user. (required)
        :return: User
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.users_partial_update_with_http_info(
                body, id, **kwargs
            )  # noqa: E501
        else:
            (data) = self.users_partial_update_with_http_info(
                body, id, **kwargs
            )  # noqa: E501
            return data

    def users_partial_update_with_http_info(
        self, body, id, **kwargs
    ):  # noqa: E501
        """Method updates chosen fields of a user  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.users_partial_update_with_http_info(body, id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param User body: (required)
        :param int id: A unique integer value identifying this user. (required)
        :return: User
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
                    " to method users_partial_update" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'body' is set
        if "body" not in params or params["body"] is None:
            raise ValueError(
                "Missing the required parameter `body` when calling `users_partial_update`"
            )  # noqa: E501
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `users_partial_update`"
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
            "/users/{id}",
            "PATCH",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="User",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def users_read(self, id, **kwargs):  # noqa: E501
        """Method provides information of a specific user  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.users_read(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this user. (required)
        :return: User
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.users_read_with_http_info(id, **kwargs)  # noqa: E501
        else:
            (data) = self.users_read_with_http_info(id, **kwargs)  # noqa: E501
            return data

    def users_read_with_http_info(self, id, **kwargs):  # noqa: E501
        """Method provides information of a specific user  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.users_read_with_http_info(id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int id: A unique integer value identifying this user. (required)
        :return: User
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
                    " to method users_read" % key
                )
            params[key] = val
        del params["kwargs"]
        # verify the required parameter 'id' is set
        if "id" not in params or params["id"] is None:
            raise ValueError(
                "Missing the required parameter `id` when calling `users_read`"
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
            "/users/{id}",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="User",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )

    def users_self(self, **kwargs):  # noqa: E501
        """Method returns an instance of a user who is currently authorized  # noqa: E501

        Method returns an instance of a user who is currently authorized  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.users_self(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str search: A search term.
        :param float id:
        :param str is_active:
        :param str ordering: Which field to use when ordering the results.
        :param int page: A page number within the paginated result set.
        :param int page_size: Number of results to return per page.
        :return: InlineResponse2002
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs["_return_http_data_only"] = True
        if kwargs.get("async_req"):
            return self.users_self_with_http_info(**kwargs)  # noqa: E501
        else:
            (data) = self.users_self_with_http_info(**kwargs)  # noqa: E501
            return data

    def users_self_with_http_info(self, **kwargs):  # noqa: E501
        """Method returns an instance of a user who is currently authorized  # noqa: E501

        Method returns an instance of a user who is currently authorized  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.users_self_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str search: A search term.
        :param float id:
        :param str is_active:
        :param str ordering: Which field to use when ordering the results.
        :param int page: A page number within the paginated result set.
        :param int page_size: Number of results to return per page.
        :return: InlineResponse2002
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = [
            "search",
            "id",
            "is_active",
            "ordering",
            "page",
            "page_size",
        ]  # noqa: E501
        all_params.append("async_req")
        all_params.append("_return_http_data_only")
        all_params.append("_preload_content")
        all_params.append("_request_timeout")

        params = locals()
        for key, val in six.iteritems(params["kwargs"]):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method users_self" % key
                )
            params[key] = val
        del params["kwargs"]

        collection_formats = {}

        path_params = {}

        query_params = []
        if "search" in params:
            query_params.append(("search", params["search"]))  # noqa: E501
        if "id" in params:
            query_params.append(("id", params["id"]))  # noqa: E501
        if "is_active" in params:
            query_params.append(
                ("is_active", params["is_active"])
            )  # noqa: E501
        if "ordering" in params:
            query_params.append(("ordering", params["ordering"]))  # noqa: E501
        if "page" in params:
            query_params.append(("page", params["page"]))  # noqa: E501
        if "page_size" in params:
            query_params.append(
                ("page_size", params["page_size"])
            )  # noqa: E501

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
            "/users/self",
            "GET",
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type="InlineResponse2002",  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get("async_req"),
            _return_http_data_only=params.get("_return_http_data_only"),
            _preload_content=params.get("_preload_content", True),
            _request_timeout=params.get("_request_timeout"),
            collection_formats=collection_formats,
        )