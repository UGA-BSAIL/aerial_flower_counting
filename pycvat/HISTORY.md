### v0.1.0 (2020-08-27)

- Initial project

### v0.2.0 (2020-09-02)

- MVP with working CLI.

### v0.2.1 (2020-09-02)

- Speed up image processing.

### v0.3.0 (2020-09-09)

- Add support for Kedro datasets.

### v0.3.1 (2020-09-20)

- Add support for getting the frame size.
- Get coverage to 100%.

### v0.3.2 (2020-10-16)

- Switch to backported version of Py 3.8 cached_property

### v0.4.0 (2020-10-19)

- Add support for creating new tasks.

### v0.5.0 (2020-10-21)

- Add support for label annotations.
- Better API for new task creation.

### v0.6.0 (2020-10-21)

- Add support for a separate authentication dataset.

### v0.6.1 (2020-10-21)

- Fix a missing import alias.

### v0.7.0 (2020-10-27)

- Completely rewrite the CVAT connector layer.
- No longer uses the CVAT CLI and `datamuro` internally.
- Connects directly to CVAT using swagger-generated client.

### v0.7.1 (2021-03-09)

- Allow newer kedro versions.

### v0.7.2 (2021-03-14)

- Make Pillow dependency less restrictive.

### v0.8.0 (2021-07-08)

- Add Task API method for getting number of frames.
