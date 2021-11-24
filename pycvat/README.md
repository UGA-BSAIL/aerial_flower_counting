# PyCvat

# Regenerating the OpenAPI Client

To do this, it is necessary that you have downloaded the
[swagger-code-gen](https://github.com/swagger-api/swagger-codegen/tree/3.0.0) 3.X tool.

1. Update `code_gen/config.json` to reflect the proper versioning for the generated package.
2. Run the following command: `java -jar swagger-codegen-cli.jar generate -i code_gen/openapi.json -l python -o python-client/ -c code_gen/config.json`
