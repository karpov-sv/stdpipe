Object flags
============

The objects detected and measured by *STDPipe* may have various flags that characterize their quality and any issues encountered during detection or measurement. These flags are combined using bitwise OR operations, allowing multiple conditions to be recorded simultaneously.

Detection Flags
---------------

**SExtractor object detection flags**

Set by :func:`~stdpipe.photometry.get_objects_sextractor`. See `SExtractor documentation <https://sextractor.readthedocs.io/en/latest/Flagging.html#extraction-flags-flags>`_ for more details.

- `0x0001` - Aperture flux is significantly affected by nearby stars or bad pixels
- `0x0002` - Object is deblended
- `0x0004` - Object is saturated
- `0x0008` - Object footprint is truncated
- `0x0010` - Object aperture data are incomplete
- `0x0020` - Object isophotal data are incomplete

**SEP object detection flags**

Set by :func:`~stdpipe.photometry.get_objects_sep`. SEP uses different flag definitions than SExtractor. See `SEP documentation <https://sep.readthedocs.io/en/v1.1.x/reference.html>`_ for details on testing flags (e.g., ``(flags & sep.OBJ_MERGED) != 0``).

SEP defines the following flag constants (exact hex values may depend on the SEP library version and are not explicitly listed in the documentation; but we list the current values here for the ease of use):

- `0x0001` - ``sep.OBJ_MERGED`` - Object is result of deblending
- `0x0002` - ``sep.OBJ_TRUNC`` - Object is truncated at image boundary
- `0x0008` - ``sep.OBJ_SINGU`` - x, y fully correlated in object
- `0x0010` - ``sep.APER_TRUNC`` - Aperture truncated at image boundary
- `0x0020` - ``sep.APER_HASMASKED`` - Aperture contains one or more masked pixels
- `0x0040` - ``sep.APER_ALLMASKED`` - Aperture contains only masked pixels
- `0x0080` - ``sep.APER_NONPOSITIVE`` - Aperture sum is negative in Kron radius

**photutils object detection flags**

Set by :func:`~stdpipe.photometry.get_objects_photutils`. The following flags can be set during detection:

- `0x0002` - Object is deblended (segmentation method only, when ``deblend=True``)
- `0x0004` - Object is saturated (when ``saturation`` parameter is provided)
- `0x0008` - Object footprint is truncated at image boundary
- `0x0010` - Object has poor quality metrics (StarFinder methods: sharpness/roundness near rejection thresholds)

**Masked pixel flag (common to detection and measurement)**

- `0x0100` - Object footprint contains masked pixels (SExtractor and other methods, or photutils when mask provided)

Measurement Flags
-----------------

**Aperture photometry flags**

Set by :func:`~stdpipe.photometry.measure_objects` during aperture photometry measurements.

- `0x0200` - Photometric aperture contains masked pixels
- `0x0400` - Local background annulus does not have enough good pixels

**Optimal extraction flags**

Set by :func:`~stdpipe.photometry.measure_objects` when using optimal extraction (Naylor 1998 algorithm).

- `0x0800` - Optimal extraction failed (NaN or non-finite flux result)

PSF Photometry Flags
--------------------

Set by :func:`~stdpipe.photometry_psf.measure_objects_psf` (photutils backend) or :func:`~stdpipe.photometry_iraf.measure_objects_iraf` (DAOPHOT backend).

- `0x1000` - PSF fit failed (NaN or non-finite flux result)
- `0x2000` - Large centroid shift during fitting (>1 pixel from initial position). Only set when centroid refinement is enabled.

Using Flags
-----------

**Checking for specific flags:**

To check if a flag is set::

    # Check for masked pixels
    masked_objs = obj[(obj['flags'] & 0x100) != 0]

    # Check for successful measurement
    good_objs = obj[(obj['flags'] & 0x100) == 0]

    # Check for multiple conditions
    unflagged = obj[(obj['flags'] & (0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000)) == 0]

**Filtering in pipeline:**

The :func:`~stdpipe.pipeline.filter_transient_candidates` function uses a default flag mask ``0x7F00`` to filter objects with measurement quality issues.

**SEP-specific flag checking:**

For objects from :func:`~stdpipe.photometry.get_objects_sep`, use SEP constants for compatibility::

    import sep

    # Check for merged objects
    unmerged = obj[(obj['flags'] & sep.OBJ_MERGED) == 0]

    # Check for complete apertures
    complete_aperture = obj[(obj['flags'] & sep.APER_TRUNC) == 0]

See Also
--------

- :doc:`detection` - Object detection documentation
- `SExtractor documentation <https://sextractor.readthedocs.io/en/latest/Flagging.html>`_
- `SEP documentation <https://sep.readthedocs.io/en/v1.1.x/>`_
