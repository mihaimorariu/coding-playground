    segment .text
    global main

main:
    push rbp
    mov  rbp, rsp
    sub  rsp, 16

    mov  rax, 0x12345678
    mov  rbx, rax
    and  rbx, 0xf     ; rbx has the low nibble 0x8
    mov  rdx, 0       ; prepare to divide
    mov  rcx, 16      ; by 16
    idiv rcx          ; rax has 0x1234567
    and  rax, 0xf     ; rax has the nibble 0x7

    leave
    ret
