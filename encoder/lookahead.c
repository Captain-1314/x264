/*****************************************************************************
 * lookahead.c: high-level lookahead functions
 *****************************************************************************
 * Copyright (C) 2010-2023 Avail Media and x264 project
 *
 * Authors: Michael Kazmier <mkazmier@availmedia.com>
 *          Alex Giladi <agiladi@availmedia.com>
 *          Steven Walters <kemuri9@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at licensing@x264.com.
 *****************************************************************************/

/* LOOKAHEAD (threaded and non-threaded mode)
 *
 * Lookahead types:
 *     [1] Slice type / scene cut;
 *
 * In non-threaded mode, we run the existing slicetype decision code as it was.
 * In threaded mode, we run in a separate thread, that lives between the calls
 * to x264_encoder_open() and x264_encoder_close(), and performs lookahead for
 * the number of frames specified in rc_lookahead.  Recommended setting is
 * # of bframes + # of threads.
 */
#include "common/common.h"
#include "analyse.h"

static void lookahead_shift( x264_sync_frame_list_t *dst, x264_sync_frame_list_t *src, int count )
{
    int i = count;
    while( i-- )
    {
        assert( dst->i_size < dst->i_max_size );
        assert( src->i_size );
        dst->list[ dst->i_size++ ] = x264_frame_shift( src->list );
        src->i_size--;
    }
    if( count )
    {
        x264_pthread_cond_broadcast( &dst->cv_fill );
        x264_pthread_cond_broadcast( &src->cv_empty );
    }
}

static void lookahead_update_last_nonb( x264_t *h, x264_frame_t *new_nonb )
{
    if( h->lookahead->last_nonb )
        x264_frame_push_unused( h, h->lookahead->last_nonb );
    h->lookahead->last_nonb = new_nonb;
    new_nonb->i_reference_count++;
}

#if HAVE_THREAD
static void lookahead_slicetype_decide( x264_t *h )
{
    x264_slicetype_decide( h );//对当前帧进行切片类型的决策

    lookahead_update_last_nonb( h, h->lookahead->next.list[0] );//更新最后一个非 B 帧（非 B-frame）的信息，参数为预测缓冲区的下一步帧列表的第一个帧
    int shift_frames = h->lookahead->next.list[0]->i_bframes + 1;//计算需要移动的帧数 shift_frames，取下一步帧列表的第一个帧的 B 帧数加1
    //锁定输出帧缓冲区的互斥锁
    x264_pthread_mutex_lock( &h->lookahead->ofbuf.mutex );
    while( h->lookahead->ofbuf.i_size == h->lookahead->ofbuf.i_max_size )//当输出帧缓冲区的大小达到最大值 h->lookahead->ofbuf.i_max_size 时，等待空闲条件变量的信号
        x264_pthread_cond_wait( &h->lookahead->ofbuf.cv_empty, &h->lookahead->ofbuf.mutex );
    //锁定下一步帧列表的互斥锁
    x264_pthread_mutex_lock( &h->lookahead->next.mutex );
    lookahead_shift( &h->lookahead->ofbuf, &h->lookahead->next, shift_frames );//将帧数据从下一步帧列表移动到输出帧缓冲区，移动的帧数为 shift_frames
    x264_pthread_mutex_unlock( &h->lookahead->next.mutex );//解锁下一步帧列表的互斥锁
    //如果预测缓冲区启用了关键帧分析（MB-tree和VBV lookahead），并且最后一个非 B 帧为 I 帧（帧类型为 I-frame）
    /* For MB-tree and VBV lookahead, we have to perform propagation analysis on I-frames too. */
    if( h->lookahead->b_analyse_keyframe && IS_X264_TYPE_I( h->lookahead->last_nonb->i_type ) )
        x264_slicetype_analyse( h, shift_frames );

    x264_pthread_mutex_unlock( &h->lookahead->ofbuf.mutex );//解锁输出帧缓冲区的互斥锁
}

REALIGN_STACK static void *lookahead_thread( x264_t *h )
{
    while( 1 )//进入一个无限循环，表示预测缓冲区线程的运行
    {
        x264_pthread_mutex_lock( &h->lookahead->ifbuf.mutex );//锁定输入帧列表的互斥锁
        if( h->lookahead->b_exit_thread )//检查是否需要退出线程
        {
            x264_pthread_mutex_unlock( &h->lookahead->ifbuf.mutex );
            break;
        }
        x264_pthread_mutex_lock( &h->lookahead->next.mutex );//锁定下一步帧列表的互斥锁
        int shift = X264_MIN( h->lookahead->next.i_max_size - h->lookahead->next.i_size, h->lookahead->ifbuf.i_size );//计算需要移动的帧数 shift，取预测缓冲区的帧数和下一步帧列表的剩余空间的较小值
        lookahead_shift( &h->lookahead->next, &h->lookahead->ifbuf, shift );//调用 lookahead_shift 函数将帧数据从输入帧列表移动到下一步帧列表
        x264_pthread_mutex_unlock( &h->lookahead->next.mutex );//解锁下一步帧列表的互斥锁
        if( h->lookahead->next.i_size <= h->lookahead->i_slicetype_length + h->param.b_vfr_input )
        {//检查下一个、步帧列表的大小是否小于等于切片类型长度加上参数 b_vfr_input
            while( !h->lookahead->ifbuf.i_size && !h->lookahead->b_exit_thread )//在输入帧列表的大小为零且线程不需要退出的情况下，等待填充条件变量的信号
                x264_pthread_cond_wait( &h->lookahead->ifbuf.cv_fill, &h->lookahead->ifbuf.mutex );
            x264_pthread_mutex_unlock( &h->lookahead->ifbuf.mutex );
        }
        else
        {   //如果下一个帧列表的大小大于切片类型长度加上参数 b_vfr_input
            x264_pthread_mutex_unlock( &h->lookahead->ifbuf.mutex );
            lookahead_slicetype_decide( h );//进行切片类型的决策
        }
    }   /* end of input frames */
    x264_pthread_mutex_lock( &h->lookahead->ifbuf.mutex );//再次锁定输入帧列表和下一步帧列表的互斥锁
    x264_pthread_mutex_lock( &h->lookahead->next.mutex );//调用 lookahead_shift 函数将剩余的输入帧移动到下一步帧列表
    lookahead_shift( &h->lookahead->next, &h->lookahead->ifbuf, h->lookahead->ifbuf.i_size );
    x264_pthread_mutex_unlock( &h->lookahead->next.mutex );//解锁下一步帧列表的互斥锁
    x264_pthread_mutex_unlock( &h->lookahead->ifbuf.mutex );//解锁输入帧列表的互斥锁
    while( h->lookahead->next.i_size )
        lookahead_slicetype_decide( h );//当下一步帧列表仍有帧时，执行切片类型的决策，即调用 lookahead_slicetype_decide 函数
    x264_pthread_mutex_lock( &h->lookahead->ofbuf.mutex );//锁定输出帧缓冲区的互斥锁
    h->lookahead->b_thread_active = 0;//将预测缓冲区的线程活跃标志 h->lookahead->b_thread_active 设置为0
    x264_pthread_cond_broadcast( &h->lookahead->ofbuf.cv_fill );//广播填充条件变量的信号，以通知等待该变量的线程可以继续执行
    x264_pthread_mutex_unlock( &h->lookahead->ofbuf.mutex );//解锁输出帧缓冲区的互斥锁
    return NULL;
}

#endif

int x264_lookahead_init( x264_t *h, int i_slicetype_length )
{
    x264_lookahead_t *look;
    CHECKED_MALLOCZERO( look, sizeof(x264_lookahead_t) );
    for( int i = 0; i < h->param.i_threads; i++ )
        h->thread[i]->lookahead = look;

    look->i_last_keyframe = - h->param.i_keyint_max;
    look->b_analyse_keyframe = (h->param.rc.b_mb_tree || (h->param.rc.i_vbv_buffer_size && h->param.rc.i_lookahead))
                               && !h->param.rc.b_stat_read;
    look->i_slicetype_length = i_slicetype_length;

    /* init frame lists */
    if( x264_sync_frame_list_init( &look->ifbuf, h->param.i_sync_lookahead+3 ) ||
        x264_sync_frame_list_init( &look->next, h->frames.i_delay+3 ) ||
        x264_sync_frame_list_init( &look->ofbuf, h->frames.i_delay+3 ) )
        goto fail;

    if( !h->param.i_sync_lookahead )
        return 0;

    x264_t *look_h = h->thread[h->param.i_threads];
    *look_h = *h;
    if( x264_macroblock_cache_allocate( look_h ) )
        goto fail;

    if( x264_macroblock_thread_allocate( look_h, 1 ) < 0 )
        goto fail;

    if( x264_pthread_create( &look->thread_handle, NULL, (void*)lookahead_thread, look_h ) )
        goto fail;
    look->b_thread_active = 1;

    return 0;
fail:
    x264_free( look );
    return -1;
}

void x264_lookahead_delete( x264_t *h )
{
    if( h->param.i_sync_lookahead )
    {
        x264_pthread_mutex_lock( &h->lookahead->ifbuf.mutex );
        h->lookahead->b_exit_thread = 1;
        x264_pthread_cond_broadcast( &h->lookahead->ifbuf.cv_fill );
        x264_pthread_mutex_unlock( &h->lookahead->ifbuf.mutex );
        x264_pthread_join( h->lookahead->thread_handle, NULL );
        x264_macroblock_cache_free( h->thread[h->param.i_threads] );
        x264_macroblock_thread_free( h->thread[h->param.i_threads], 1 );
        x264_free( h->thread[h->param.i_threads] );
    }
    x264_sync_frame_list_delete( &h->lookahead->ifbuf );
    x264_sync_frame_list_delete( &h->lookahead->next );
    if( h->lookahead->last_nonb )
        x264_frame_push_unused( h, h->lookahead->last_nonb );
    x264_sync_frame_list_delete( &h->lookahead->ofbuf );
    x264_free( h->lookahead );
}

void x264_lookahead_put_frame( x264_t *h, x264_frame_t *frame )
{   //如果存在预测线程，则调用 x264_sync_frame_list_push 函数，将帧数据 frame 推入预测缓冲区的输入帧列表
    if( h->param.i_sync_lookahead )
        x264_sync_frame_list_push( &h->lookahead->ifbuf, frame );
    else//如果不存在预测线程，则调用 x264_sync_frame_list_push 函数，将帧数据 frame 推入预测缓冲区的下一个帧列表
        x264_sync_frame_list_push( &h->lookahead->next, frame );
}

int x264_lookahead_is_empty( x264_t *h )
{
    x264_pthread_mutex_lock( &h->lookahead->ofbuf.mutex );
    x264_pthread_mutex_lock( &h->lookahead->next.mutex );
    int b_empty = !h->lookahead->next.i_size && !h->lookahead->ofbuf.i_size;
    x264_pthread_mutex_unlock( &h->lookahead->next.mutex );
    x264_pthread_mutex_unlock( &h->lookahead->ofbuf.mutex );
    return b_empty;
}

static void lookahead_encoder_shift( x264_t *h )
{
    if( !h->lookahead->ofbuf.i_size )
        return;
    int i_frames = h->lookahead->ofbuf.list[0]->i_bframes + 1;
    while( i_frames-- )
    {
        x264_frame_push( h->frames.current, x264_frame_shift( h->lookahead->ofbuf.list ) );
        h->lookahead->ofbuf.i_size--;
    }
    x264_pthread_cond_broadcast( &h->lookahead->ofbuf.cv_empty );
}

void x264_lookahead_get_frames( x264_t *h )
{
    if( h->param.i_sync_lookahead )
    {   /* We have a lookahead thread, so get frames from there */
        x264_pthread_mutex_lock( &h->lookahead->ofbuf.mutex );
        while( !h->lookahead->ofbuf.i_size && h->lookahead->b_thread_active )
            x264_pthread_cond_wait( &h->lookahead->ofbuf.cv_fill, &h->lookahead->ofbuf.mutex );
        lookahead_encoder_shift( h );
        x264_pthread_mutex_unlock( &h->lookahead->ofbuf.mutex );
    }
    else
    {   /* We are not running a lookahead thread, so perform all the slicetype decide on the fly */

        if( h->frames.current[0] || !h->lookahead->next.i_size )
            return;

        x264_slicetype_decide( h );
        lookahead_update_last_nonb( h, h->lookahead->next.list[0] );
        int shift_frames = h->lookahead->next.list[0]->i_bframes + 1;
        lookahead_shift( &h->lookahead->ofbuf, &h->lookahead->next, shift_frames );

        /* For MB-tree and VBV lookahead, we have to perform propagation analysis on I-frames too. */
        if( h->lookahead->b_analyse_keyframe && IS_X264_TYPE_I( h->lookahead->last_nonb->i_type ) )
            x264_slicetype_analyse( h, shift_frames );

        lookahead_encoder_shift( h );
    }
}
