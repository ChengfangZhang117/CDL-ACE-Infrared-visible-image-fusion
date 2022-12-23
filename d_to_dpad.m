function dpad = d_to_dpad(d, size_xpad, size_kernel, center)

%pad filters (for multiple filters)
dpad = padarray( d, [size_xpad(1)-size_kernel(1), size_xpad(2)-size_kernel(2), 0], 0, 'post' );
dpad = circshift(dpad, [1-center(1), 1-center(2), 0]);

return;