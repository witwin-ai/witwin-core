# Reviewed core source integration

The clean core main branch was fast-forwarded from78aad43 to the already
locally tracked0.4 release46c826d. All14 installed0.4 source files match that
release byte-for-byte, so this aligns source with the active runtime instead
of replacing it with the old0.3 API. Original wheel build commit metadata is
absent; the content comparison establishes source equivalence only.

The preferred one-file nested-Tensor fix and49 focused contracts were then
applied unchanged. All current core CPU tests passed:150 passed, four CUDA
cases deferred. The private baseline has39 real failures on those49 focused
contracts; the corrected candidate passes49. Maxwell CPU Scene/material
compiler integration passes101. Actual source/controller hashes and old
failures remain in the copied evidence. GPU tests and a complete final Maxwell
regression against the new installed artifact remain necessary.
