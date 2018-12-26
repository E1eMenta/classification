import torch
import torch.nn.functional as F

from pytrainer.utils import AverageMeter



class ClassificationValidator:
    def __init__(
            self,
            DataLoader,
            save_best=True,
            top5=False,
            tag="",
    ):
        self.tag = tag
        self.save_best = save_best
        self.top5 = top5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataLoader = DataLoader

        self.best_accuracy = 0

    def __call__(self, model, params):

        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        for data, target in self.dataLoader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            loss = F.cross_entropy(output, target)  # sum up batch loss

            prec1, prec5 = self.get_accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

        self.accuracy = top1.avg.item()
        self.top5 = top5.avg.item()

        iteration = params["batch"]
        if self.accuracy > self.best_accuracy:
            self.best_accuracy = self.accuracy

            if self.save_best:
                save_path = params["save_dir"]+ "/iter_{}_acc{:.2f}.pth".format(iteration, self.accuracy)
                if model.__class__.__name__ == "DataParallel":
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(),        save_path)

        report_line = 'Test: Average loss: {:.2f}, Accuracy: {:.2f}%'.format(losses.avg, self.accuracy)
        if self.top5:
            report_line += ' Top5: {:.2f}%'.format(top5.avg.item())
        print(report_line)

        writer = params["writer"]
        writer.add_scalar('val_loss', losses.avg, iteration)
        writer.add_scalar('val_accuracy', self.accuracy, iteration)
        if self.top5:
            writer.add_scalar('top5', top5.avg, iteration)

        top1.reset()
        top5.reset()
        losses.reset()

    # Get from https://github.com/pytorch/examples/blob/master/imagenet
    def get_accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res