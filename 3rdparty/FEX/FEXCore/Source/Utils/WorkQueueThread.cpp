// SPDX-License-Identifier: MIT
#include <FEXCore/Utils/WorkQueueThread.h>
#include <FEXCore/Utils/LogManager.h>

namespace FEXCore {

WorkQueueThread::WorkQueueThread(FEXCore::Threads::Flags ThreadFlags, const char* ThreadName) {
  Thread = FEXCore::Threads::Thread::Create(ThreadEntry, this, ThreadFlags, ThreadName);
}

WorkQueueThread::~WorkQueueThread() {
  {
    std::unique_lock lk {Mutex};
    Stop = true;
  }
  CV.notify_one();

  if (Thread && Thread->joinable()) {
    Thread->join(nullptr);
  }
}

void WorkQueueThread::QueueWork(fextl::unique_ptr<WorkItem> Work) {
  {
    std::unique_lock lk {Mutex};
    Queue.push_back(std::move(Work));
  }
  CV.notify_one();
}

void WorkQueueThread::WaitIdle() {
  std::unique_lock lk {Mutex};
  IdleCV.wait(lk, [this] { return Queue.empty() && !Running; });
}

void WorkQueueThread::ThreadProc() {
  while (true) {
    fextl::unique_ptr<WorkItem> Work;
    {
      std::unique_lock lk {Mutex};
      while (!(Stop || !Queue.empty())) {
        CV.wait(lk);
      }
      if (Queue.empty()) {
        // nothing to do? must be stopping
        LOGMAN_THROW_A_FMT(Stop, "WorkQueueThread wakes up empty but no Stop?");
        return;
      }

      Work = std::move(Queue.front());
      Queue.pop_front();
      Running = true;
    }

    Work->Run();
    Work.reset();
    {
      std::unique_lock lk {Mutex};
      Running = false;
    }
    IdleCV.notify_all();
    // Work is destroyed here
  }
}

} // namespace FEXCore
