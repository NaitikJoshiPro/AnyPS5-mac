// SPDX-License-Identifier: MIT
#pragma once

#include <FEXCore/Utils/Threads.h>
#include <FEXCore/fextl/deque.h>
#include <FEXCore/fextl/memory.h>

#include <condition_variable>
#include <mutex>

namespace FEXCore {

class WorkQueueThread {
public:
  // destroyed after Run()
  struct WorkItem {
    virtual ~WorkItem() = default;
    virtual void Run() = 0;
  };

  WorkQueueThread(FEXCore::Threads::Flags Flags = {}, const char* ThreadName = nullptr);
  ~WorkQueueThread();

  void QueueWork(fextl::unique_ptr<WorkItem> Work);

  // Blocks until every queued item has run.
  void WaitIdle();

private:
  // static function for the ::Thread to refer to
  static void* ThreadEntry(void* Self) {
    static_cast<WorkQueueThread*>(Self)->ThreadProc();
    return nullptr;
  }
  void ThreadProc();

  std::mutex Mutex;
  std::condition_variable CV;
  std::condition_variable IdleCV;
  bool Running = false;
  fextl::deque<fextl::unique_ptr<WorkItem>> Queue;
  bool Stop = false;

  fextl::unique_ptr<FEXCore::Threads::Thread> Thread;
};
} // namespace FEXCore
