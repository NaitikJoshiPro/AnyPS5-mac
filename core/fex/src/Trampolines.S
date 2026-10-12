    .text

    .globl _Aps5NativeCall
    .p2align 2
_Aps5NativeCall:
    .cfi_startproc
    stp x29, x30, [sp, #-32]!
    .cfi_def_cfa_offset 32
    .cfi_offset w30, -24
    .cfi_offset w29, -32
    mov x29, sp
    .cfi_def_cfa w29, 32
    stp x19, x20, [sp, #16]
    .cfi_offset w20, -8
    .cfi_offset w19, -16
    mov x19, x2
    mov x16, x0
    mov x17, x1
    ldr x10, [x17, #192]
    ldr x9, [x17, #200]
    lsl x11, x9, #3
    add x11, x11, #15
    and x11, x11, #~15
    sub sp, sp, x11
    mov x12, sp
1:
    cbz x9, 2f
    ldr x13, [x10], #8
    str x13, [x12], #8
    sub x9, x9, #1
    b 1b
2:
    ldp q0, q1, [x17, #64]
    ldp q2, q3, [x17, #96]
    ldp q4, q5, [x17, #128]
    ldp q6, q7, [x17, #160]
    ldp x0, x1, [x17, #0]
    ldp x2, x3, [x17, #16]
    ldp x4, x5, [x17, #32]
    ldp x6, x7, [x17, #48]
    blr x16
    stp x0, x1, [x19, #0]
    stp q0, q1, [x19, #16]
    mov sp, x29
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret
    .cfi_endproc

    .globl _Aps5GuestCallEntry
    .p2align 2
_Aps5GuestCallEntry:
    .cfi_startproc
    stp x29, x30, [sp, #-16]!
    .cfi_def_cfa_offset 16
    .cfi_offset w30, -8
    .cfi_offset w29, -16
    mov x29, sp
    .cfi_def_cfa w29, 16
    sub sp, sp, #192
    stp x0, x1, [sp, #0]
    stp x2, x3, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp q0, q1, [sp, #64]
    stp q2, q3, [sp, #96]
    stp q4, q5, [sp, #128]
    stp q6, q7, [sp, #160]
    mov x0, x16
    mov x1, sp
    bl _Aps5RunGuestCall
    ldp x0, x1, [sp, #0]
    ldp q0, q1, [sp, #64]
    mov sp, x29
    ldp x29, x30, [sp], #16
    ret
    .cfi_endproc
