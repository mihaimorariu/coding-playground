    segment .data
a   dd      4
b   dd      4.4
c   times   10 dd 0
d   dw      1, 2
e   db      0xfb
f   db      "hello world", 0

    segment .bss
g   resd    1
h   resd    10
i   resb    100

    segment .text
    global  main

main:
    push    rbp
    mov     rbp, rsp
    sub     rsp, 16

    xor     eax, eax
    leave
    ret
