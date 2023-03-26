    segment .text
    global main

main:
    push rbp
    mov  rbp, rsp
    sub  rsp, 16

    mov  rax, 0
    not  rax
    mov  rdx, 0
    mov  rbx, 15
    div  rbx
    not  rax

    leave
    ret
